/* Experimental native batch query over independent M-DCI instances.
 *
 * Build only against the exact M-DCI headers used for the installed dciknn
 * extension.  The capsule layout below mirrors upstream src/py_dci.c.
 * No Python API is used while the GIL is released in the OpenMP region.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include "dci.h"
#include "hashtable_pp.h"
#include "btree_i.h"

typedef struct {
    dci dci_inst;
    hashtable_pp hashtable;
    btree_i *cached_tree;
} py_dci;
typedef struct {
    py_dci *dci_inst_list;
    int num_inst;
} py_dci_list;

typedef struct {
    py_dci_list *db;
    PyArrayObject *query;
    PyArrayObject *output;
    int neighbours;
    int field_of_view;
    int ratio;
} request;

static int last_team_size = 0;
static int last_task_count = 0;

static PyObject *batch_query(PyObject *unused, PyObject *args) {
    PyObject *capsules, *queries, *neighbours, *fields;
    int ratio, threads;
    if (!PyArg_ParseTuple(args, "OOOOii", &capsules, &queries, &neighbours,
                          &fields, &ratio, &threads)) return NULL;
    PyObject *cs = PySequence_Fast(capsules, "capsules must be a sequence");
    PyObject *qs = PySequence_Fast(queries, "queries must be a sequence");
    PyObject *ns = PySequence_Fast(neighbours, "neighbours must be a sequence");
    PyObject *fs = PySequence_Fast(fields, "fields must be a sequence");
    if (!cs || !qs || !ns || !fs) goto fail_sequences;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(cs);
    if (n < 1 || PySequence_Fast_GET_SIZE(qs) != n ||
        PySequence_Fast_GET_SIZE(ns) != n || PySequence_Fast_GET_SIZE(fs) != n ||
        ratio < 1 || threads < 1) {
        PyErr_SetString(PyExc_ValueError, "invalid batch query dimensions");
        goto fail_sequences;
    }
    request *req = calloc((size_t)n, sizeof(request));
    PyObject *results = PyList_New(n);
    if (!req || !results) {
        PyErr_NoMemory();
        free(req);
        Py_XDECREF(results);
        goto fail_sequences;
    }
    int total_heads = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        req[i].db = PyCapsule_GetPointer(PySequence_Fast_GET_ITEM(cs, i),
                                          "py_dci_inst_list");
        if (!req[i].db) goto fail_requests;
        PyArrayObject *q = (PyArrayObject *)PyArray_FROM_OTF(
            PySequence_Fast_GET_ITEM(qs, i), NPY_FLOAT32, NPY_ARRAY_CARRAY_RO);
        if (!q) goto fail_requests;
        req[i].query = q;
        req[i].neighbours = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(ns, i));
        req[i].field_of_view = (int)PyLong_AsLong(PySequence_Fast_GET_ITEM(fs, i));
        if (PyErr_Occurred()) goto fail_requests;
        req[i].ratio = ratio;
        int heads = req[i].db->num_inst * ratio;
        int dim = req[i].db->dci_inst_list[0].dci_inst.dim;
        if (heads < 1 || dim < 1 || req[i].neighbours < 1 ||
            PyArray_NDIM(q) != 2 || PyArray_DIM(q, 0) != heads ||
            PyArray_DIM(q, 1) != dim) {
            PyErr_SetString(PyExc_ValueError, "invalid DCI capsule or query shape");
            goto fail_requests;
        }
        for (int h = 0; h < req[i].db->num_inst; ++h)
            if (req[i].db->dci_inst_list[h].dci_inst.num_points < req[i].neighbours) {
                PyErr_SetString(PyExc_ValueError,
                                "requested neighbours exceed a tree's point count");
                goto fail_requests;
            }
        npy_intp shape = (npy_intp)heads * req[i].neighbours * 2;
        req[i].output = (PyArrayObject *)PyArray_SimpleNew(1, &shape, NPY_INT32);
        if (!req[i].output) goto fail_requests;
        memset(PyArray_DATA(req[i].output), 0xff, (size_t)shape * sizeof(int));
        Py_INCREF(req[i].output);
        PyList_SET_ITEM(results, i, (PyObject *)req[i].output);
        total_heads += heads;
    }
    int *returned = malloc((size_t)total_heads * sizeof(int));
    if (!returned) {
        PyErr_NoMemory();
        goto fail_requests;
    }

    /* One fixed team for the entire batch; each task is one request/head.
       dci_query sees parallel_level=0, so it does not spawn nested teams. */
    int actual_team_size = 0;
    Py_BEGIN_ALLOW_THREADS
#pragma omp parallel for num_threads(threads) schedule(dynamic)
    for (int flat = 0; flat < total_heads; ++flat) {
        if (flat == 0) actual_team_size = omp_get_num_threads();
        int r = 0, h = flat;
        while (h >= req[r].db->num_inst * ratio) {
            h -= req[r].db->num_inst * ratio;
            ++r;
        }
        /* Query only reads the index.  Copy its descriptor so disabling
           nested parallelism never mutates the owning DCI instance. */
        dci tree_view = req[r].db->dci_inst_list[h / ratio].dci_inst;
        tree_view.parallel_level = 0;
        dci *tree = &tree_view;
        dci_query_config cfg = {0};
        cfg.blind = false;
        cfg.num_to_visit = tree->num_points;
        cfg.num_to_retrieve = -1;
        cfg.prop_to_visit = 1.0f;
        cfg.prop_to_retrieve = 0.8f;
        cfg.field_of_view = req[r].db->dci_inst_list[0].dci_inst.num_levels >= 2
            ? req[r].field_of_view : -1;
        cfg.target_level = 0;
        bool mask = true;
        int *nearest[1] = {NULL};
        int count = 0;
        const float *query = (const float *)PyArray_DATA(req[r].query) + (size_t)h * tree->dim;
        dci_query(tree, tree->dim, 1, query, req[r].neighbours, cfg,
                  &mask, nearest, NULL, &count);
        returned[flat] = count;
        if (nearest[0]) {
            int copied = count < req[r].neighbours ? count : req[r].neighbours;
            int *out = (int *)PyArray_DATA(req[r].output) +
                       (size_t)h * req[r].neighbours * 2;
            memcpy(out, nearest[0], (size_t)copied * sizeof(int));
            memcpy(out + req[r].neighbours, nearest[0] + req[r].neighbours,
                   (size_t)copied * sizeof(int));
            free(nearest[0]);
        }
    }
    Py_END_ALLOW_THREADS
    last_team_size = actual_team_size;
    last_task_count = total_heads;

    int short_result = 0;
    int flat = 0;
    for (Py_ssize_t r = 0; r < n; ++r)
        for (int h = 0; h < req[r].db->num_inst * ratio; ++h)
            if (returned[flat++] != req[r].neighbours) short_result = 1;
    free(returned);
    for (Py_ssize_t i = 0; i < n; ++i) {
        Py_XDECREF(req[i].query);
        Py_XDECREF(req[i].output);
    }
    free(req);
    Py_DECREF(cs); Py_DECREF(qs); Py_DECREF(ns); Py_DECREF(fs);
    if (short_result) {
        Py_DECREF(results);
        PyErr_SetString(PyExc_RuntimeError,
                        "native DCI returned fewer neighbours than requested");
        return NULL;
    }
    return results;

fail_requests:
    for (Py_ssize_t i = 0; i < n; ++i) {
        Py_XDECREF(req[i].query);
        Py_XDECREF(req[i].output);
    }
    free(req);
    Py_DECREF(results);
fail_sequences:
    Py_XDECREF(cs); Py_XDECREF(qs); Py_XDECREF(ns); Py_XDECREF(fs);
    return NULL;
}

static PyObject *last_query_stats(PyObject *unused, PyObject *args) {
    return Py_BuildValue("{s:i,s:i}", "omp_team_size", last_team_size,
                         "query_tasks", last_task_count);
}

static PyMethodDef methods[] = {
    {"batch_query", batch_query, METH_VARARGS,
     "Query independent DCI trees with one fixed OpenMP worker team."},
    {"last_query_stats", last_query_stats, METH_NOARGS,
     "Diagnostics for the most recently completed native query."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_mdci_batch", NULL, -1, methods
};
PyMODINIT_FUNC PyInit__mdci_batch(void) {
    import_array();
    PyObject *result = PyModule_Create(&module);
    if (!result) return NULL;
    if (PyModule_AddStringConstant(result, "producer_sha256", ICECACHE_DCI_SHA256) < 0) {
        Py_DECREF(result);
        return NULL;
    }
    return result;
}
