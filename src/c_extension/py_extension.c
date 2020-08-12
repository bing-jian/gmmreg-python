#include <Python.h>

/*
References:
[0] https://docs.python.org/3/howto/cporting.html
[1]  Writing a C extension to NumPy
  http://folk.uio.no/inf3330/scripting/doc/python/NumPy/Numeric/numpy-13.html
[2]  How to extend NumPy
  https://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
*/

#if PY_MAJOR_VERSION >= 3
#define IS_PY3K
#endif

#include "numpy/arrayobject.h"

#define IND2(a, i, j) \
  *((double *)(a->data + i * a->strides[0] + j * a->strides[1]))

#include "cvgmi_API.h"

static PyObject *py_squared_distance_matrix(PyObject *self, PyObject *args) {
  int m, n, d;
  double *dist;
  PyObject *A, *B, *g;
  PyArrayObject *arrayA, *arrayB, *arrayg;
  PyArrayObject *array_dist;
  npy_intp out_dim[2];

  if (!PyArg_ParseTuple(args, "OOO", &A, &B, &g)) {
    return NULL;
  }

  arrayA =
      (PyArrayObject *)PyArray_ContiguousFromObject(A, PyArray_DOUBLE, 1, 2);
  arrayB =
      (PyArrayObject *)PyArray_ContiguousFromObject(B, PyArray_DOUBLE, 1, 2);
  arrayg =
      (PyArrayObject *)PyArray_ContiguousFromObject(g, PyArray_DOUBLE, 1, 2);

  if (arrayA->nd > 2 || arrayA->descr->type_num != PyArray_DOUBLE) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be two-dimensional and of type float");
    return NULL;
  }

  m = (arrayA->dimensions)[0];
  n = (arrayB->dimensions)[0];
  if (arrayA->nd > 1) {
    d = (arrayA->dimensions)[1];
  } else {
    d = 1;
  }
  dist = (double *)malloc(m * n * sizeof(double));
  squared_distance_matrix((double *)(arrayA->data), (double *)(arrayB->data),
                          (double *)(arrayg->data), m, n, d, dist);
  out_dim[0] = m;
  out_dim[1] = n;

  /*
   * PyArray_FromDimsAndData() is obsolete in NumPy 1.0 or later.
   * Use PyArray_SimpleNewFromData() instead but please be aware of the
   * potential memory leak if not used properly.
   * https://stackoverflow.com/questions/27912483/memory-leak-in-python-extension-when-array-is-created-with-pyarray-simplenewfrom
   * https://stackoverflow.com/questions/52731884/pyarray-simplenewfromdata
  */
  array_dist = (PyArrayObject*) PyArray_SimpleNewFromData(2, out_dim, NPY_DOUBLE, dist);
  PyArray_ENABLEFLAGS((PyArrayObject*) array_dist, NPY_ARRAY_OWNDATA);

  if (array_dist == NULL) {
    printf("creating %ldx%ld array failed\n", out_dim[0], out_dim[1]);
    return NULL;
  }

  Py_DECREF(arrayA);
  Py_DECREF(arrayB);
  Py_DECREF(arrayg);
  return PyArray_Return(array_dist);
}

static PyObject *py_gauss_transform(PyObject *self, PyObject *args) {
  int m, n, dim;
  double scale, result;
  double *grad;
  PyObject *A, *B;
  PyArrayObject *arrayA, *arrayB;
  PyArrayObject *arrayGrad;
  PyObject *list;

  if (!PyArg_ParseTuple(args, "OOd", &A, &B, &scale)) {
    return NULL;
  }

  arrayA =
      (PyArrayObject *)PyArray_ContiguousFromObject(A, PyArray_DOUBLE, 1, 2);
  arrayB =
      (PyArrayObject *)PyArray_ContiguousFromObject(B, PyArray_DOUBLE, 1, 2);

  if (arrayA->nd > 2 || arrayA->descr->type_num != PyArray_DOUBLE) {
    PyErr_SetString(PyExc_ValueError,
                    "array must be two-dimensional and of type float");
    return NULL;
  }

  m = (arrayA->dimensions)[0];
  n = (arrayB->dimensions)[0];
  if (arrayA->nd > 1) {
    dim = (arrayA->dimensions)[1];
  } else {
    dim = 1;
  }
  grad = (double *)malloc(m * dim * sizeof(double));
  result = GaussTransform((double *)(arrayA->data), (double *)(arrayB->data), m,
                          n, dim, scale, grad);

  /* https://stackoverflow.com/questions/52731884/pyarray-simplenewfromdata */
  arrayGrad =
      (PyArrayObject*) PyArray_SimpleNewFromData(2, arrayA->dimensions, NPY_DOUBLE, grad);
  PyArray_ENABLEFLAGS((PyArrayObject*) arrayGrad, NPY_ARRAY_OWNDATA);

  if (arrayGrad == NULL) {
    printf("creating %ldx%ld array failed\n", arrayA->dimensions[0],
           arrayA->dimensions[1]);
    return NULL;
  }

  Py_DECREF(arrayA);
  Py_DECREF(arrayB);
  list = PyList_New(0);
  if (PyList_Append(list, PyFloat_FromDouble(result)) != 0) {
    // set exception context, raise (return 0)
    return 0;
  }
  if (PyList_Append(list, PyArray_Return(arrayGrad)) != 0) {
    // set exception context, raise (return 0)
    return 0;
  }
  /* Important: arrayGrad need to be DECREFed as PyList_Append does INCREF
   * https://stackoverflow.com/questions/3512414/does-this-pylist-appendlist-py-buildvalue-leak
   * */
  Py_DECREF(arrayGrad);
  return list;
}

/*
#if PY_MAJOR_VERSION >= 3
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
#else
    #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
#endif

MODINIT(_extension)
*/

static PyMethodDef pyMethods[] = {
    {"squared_distance_matrix", py_squared_distance_matrix, METH_VARARGS,
     "Compute the squared distance matrix."},
    {"gauss_transform", py_gauss_transform, METH_VARARGS,
     "Compute the Gauss Transform."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef gmmreg_module = {
    PyModuleDef_HEAD_INIT, "_extension",
    "robust point set registration algorithm.",
    -1,  // global state
    pyMethods};

PyMODINIT_FUNC PyInit__extension(void) {
  PyObject *m;
  m = PyModule_Create(&gmmreg_module);
  if (!m) {
    return NULL;
  }
  import_array();
  return m;
}
#else

PyMODINIT_FUNC init_extension(void) {
  (void)Py_InitModule("_extension", pyMethods);
  import_array();
}

#endif
