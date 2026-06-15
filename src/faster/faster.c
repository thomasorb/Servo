#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <string.h>
/* ============================================================
   Helper macros
   ============================================================ */

static int ensure_float32_2d(PyArrayObject* a) {
    /* Ensure: 2D, float32, C‑contiguous */
    return PyArray_NDIM(a)==2 &&
           PyArray_TYPE(a)==NPY_FLOAT32 &&
           PyArray_ISCONTIGUOUS(a);
}

static int ensure_float32_1d(PyArrayObject* a) {
    /* Ensure: 1D, float32, C‑contiguous */
    return PyArray_NDIM(a)==1 &&
           PyArray_TYPE(a)==NPY_FLOAT32 &&
           PyArray_ISCONTIGUOUS(a);
}

static int ensure_int32_1d(PyArrayObject* a) {
    /* Ensure: 1D, int32, C‑contiguous */
    return PyArray_NDIM(a)==1 &&
           PyArray_TYPE(a)==NPY_INT32 &&
           PyArray_ISCONTIGUOUS(a);
}

#include <stdint.h>

static inline int is_nan_f32(float v)
{
    uint32_t u;
    memcpy(&u, &v, sizeof(float));
    return (u & 0x7fffffffU) > 0x7f800000U;
}


static inline float compute_slope(const float* x, int start, int end)
{
    /* Compute mean(diff(unwrap(x[start:end], discont=pi))) */

    int n = end - start;
    if (n < 2)
        return NPY_NAN;

    const float pi = (float)NPY_PI;
    const float two_pi = 2.0f * pi;

    float prev = x[start];
    float offset = 0.0f;

    double acc = 0.0;
    int count = 0;

    for (int i = start + 1; i < end; ++i) {
        float cur = x[i];

        /* unwrap step */
        float delta = cur - prev;
        if (delta > pi)
            offset -= two_pi;
        else if (delta < -pi)
            offset += two_pi;

        /* diff of unwrapped signal */
        float unwrapped = cur + offset;
        float prev_unwrapped = prev + offset;

        acc += (double)(unwrapped - prev_unwrapped);
        count++;

        prev = cur;
    }

    return (count > 0) ? (float)(acc / (double)count) : NPY_NAN;
}

static void replace_nan_with_min_f32(PyArrayObject *profile)
{
    float *data = (float *)PyArray_DATA(profile);
    npy_intp n = PyArray_SIZE(profile);

    float min_val = INFINITY;
    int has_valid = 0;

    /* 1) trouver le minimum (ignorer NaN) */
    for (npy_intp i = 0; i < n; ++i) {
        float v = data[i];
        if (!is_nan_f32(v)) {
            if (v < min_val) {
                min_val = v;
            }
            has_valid = 1;
        }
    }

    /* cas pathologique : tout est NaN */
    if (!has_valid) {
        for (npy_intp i = 0; i < n; ++i) {
            data[i] = 0.0f;
        }
        return;
    }

    /* 2) remplacer les NaN par le minimum */
    for (npy_intp i = 0; i < n; ++i) {
      if (is_nan_f32(data[i])) {
            data[i] = min_val;
        }
    }
}
/* ============================================================
   1) compute_profiles_local_f32
      ----------------------------------------------------------
      Computes horizontal and vertical profiles around (x,y)
      using a local (non‑integral) mean over a strip.

      Horizontal:
        hprofile[i] = mean of row r = x - hl + i
                       over columns [y-hw, y+hw)

      Vertical:
        vprofile[i] = mean of column c = y - hl + i
                       over rows [x-hw, x+hw)

      ROI:
        Returns a rectangular region [x-hl:x+hl, y-hl:y+hl]
        truncated to image boundaries.

      Requirements:
        - a: float32 2D array
        - w and l must be even
        - returns (vprofile, hprofile, roi)
   ============================================================ */

static PyObject* py_compute_slope_f32(PyObject* self, PyObject* args)
{
    PyObject *x_obj;
    int start, end;

    if (!PyArg_ParseTuple(args, "Oii", &x_obj, &start, &end))
        return NULL;

    PyArrayObject* x = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);

    if (!x)
        return NULL;

    if (!ensure_float32_1d(x)) {
        Py_DECREF(x);
        PyErr_SetString(PyExc_ValueError, "x must be a 1D float32 array");
        return NULL;
    }

    npy_intp n = PyArray_DIM(x, 0);
    if (start < 0 || end > n || start >= end) {
        Py_DECREF(x);
        PyErr_SetString(PyExc_ValueError, "invalid start/end");
        return NULL;
    }

    float* X = (float*)PyArray_DATA(x);

    float result;

    Py_BEGIN_ALLOW_THREADS
    result = compute_slope(X, start, end);
    Py_END_ALLOW_THREADS

    Py_DECREF(x);

    return PyFloat_FromDouble((double)result);
}


static PyObject* py_compute_profiles_local_f32(
    PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject *a_obj = NULL;
    PyObject *mask_obj = Py_None; /* optional weights map */
    int x, y, w, l;
    int get_roi = 1;

    /* mask is optional and must be passed as keyword (or as last positional) */
    static char* kwlist[] = {"a","x","y","w","l","get_roi","mask",NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Oiiiip|O",
            kwlist, &a_obj, &x, &y, &w, &l, &get_roi, &mask_obj))
        return NULL;

    /* Convert input image to float32 contiguous */
    PyArrayObject* a = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!a) return NULL;

    if (PyArray_NDIM(a) != 2) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_ValueError, "a must be 2D float32 array");
        return NULL;
    }

    npy_intp nrows = PyArray_DIM(a, 0);
    npy_intp ncols = PyArray_DIM(a, 1);
    float* A = (float*)PyArray_DATA(a);

    if ((w & 1) || (l & 1)) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_ValueError, "w and l must be even");
        return NULL;
    }

    int hw = w / 2;
    int hl = l / 2;

    /* Optional weight map (mask): float32 2D contiguous, same shape as a */
    PyArrayObject* mask = NULL;
    float* W = NULL;

    if (mask_obj != Py_None) {
        mask = (PyArrayObject*)PyArray_FROM_OTF(
            mask_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
        if (!mask) {
            Py_DECREF(a);
            return NULL;
        }
        if (PyArray_NDIM(mask) != 2) {
            Py_DECREF(a);
            Py_DECREF(mask);
            PyErr_SetString(PyExc_ValueError, "mask must be a 2D float32 array (weights)");
            return NULL;
        }
        if (PyArray_DIM(mask,0) != nrows || PyArray_DIM(mask,1) != ncols) {
            Py_DECREF(a);
            Py_DECREF(mask);
            PyErr_SetString(PyExc_ValueError, "mask must have the same shape as a");
            return NULL;
        }
        W = (float*)PyArray_DATA(mask);
    }

    /* Allocate output profiles */
    npy_intp dims[1] = { (npy_intp)l };
    PyArrayObject* hprof = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyArrayObject* vprof = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);

    if (!hprof || !vprof) {
        Py_XDECREF(hprof);
        Py_XDECREF(vprof);
        Py_DECREF(a);
        Py_XDECREF(mask);
        return NULL;
    }

    float* HP = (float*)PyArray_DATA(hprof);
    float* VP = (float*)PyArray_DATA(vprof);

    /* Compute ROI bounds */
    int r0 = x - hl;
    int r1 = x + hl;
    int c0 = y - hl;
    int c1 = y + hl;

    int rr0 = (r0 < 0) ? 0 : r0;
    int rr1 = (r1 > (int)nrows) ? (int)nrows : r1;
    int cc0 = (c0 < 0) ? 0 : c0;
    int cc1 = (c1 > (int)ncols) ? (int)ncols : c1;

    /* ---- Compute profiles ---- */
    Py_BEGIN_ALLOW_THREADS
    for (int i = 0; i < l; ++i) {

        /* -------- Horizontal profile -------- */
        int r = x - hl + i;
        if (r >= 0 && r < (int)nrows) {
            int c0i = (y - hw < 0) ? 0 : (y - hw);
            int c1i = (y + hw > (int)ncols) ? (int)ncols : (y + hw);
            int width = c1i - c0i;

            if (width <= 0) {
                HP[i] = NPY_NAN;
            } else {
                npy_intp base = (npy_intp)r * ncols;
                const float* row = A + base;

                double acc = 0.0;
                double wsum = 0.0;

                /* Weighted mean:
                   - if mask/weights provided: use w = W[idx]
                   - ignore pixels with w == 0
                   - ignore NaNs in v or w
                   - result = acc / wsum, and if wsum == 0 -> NaN
                */
                for (int c = c0i; c < c1i; ++c) {
                    npy_intp idx = base + (npy_intp)c;

                    float v = row[c];
                    if (is_nan_f32(v))
                        continue;

                    float wt = 1.0f;
                    if (W) {
                        wt = W[idx];
                        if (is_nan_f32(wt) || wt == 0.0f)
                            continue;
                    }

                    acc  += (double)v * (double)wt;
                    wsum += (double)wt;
                }

                HP[i] = (wsum != 0.0) ? (float)(acc / wsum) : (float)NPY_NAN;
            }
        } else {
            HP[i] = NPY_NAN;
        }

        /* -------- Vertical profile -------- */
        int c = y - hl + i;
        if (c >= 0 && c < (int)ncols) {
            int r0i = (x - hw < 0) ? 0 : (x - hw);
            int r1i = (x + hw > (int)nrows) ? (int)nrows : (x + hw);
            int height = r1i - r0i;

            if (height <= 0) {
                VP[i] = NPY_NAN;
            } else {
                double acc = 0.0;
                double wsum = 0.0;

                for (int r2 = r0i; r2 < r1i; ++r2) {
                    npy_intp idx = ((npy_intp)r2) * ncols + (npy_intp)c;

                    float v = A[idx];
                    if (is_nan_f32(v))
                        continue;

                    float wt = 1.0f;
                    if (W) {
                        wt = W[idx];
                        if (is_nan_f32(wt) || wt == 0.0f)
                            continue;
                    }

                    acc  += (double)v * (double)wt;
                    wsum += (double)wt;
                }

                VP[i] = (wsum != 0.0) ? (float)(acc / wsum) : (float)NPY_NAN;
            }
        } else {
            VP[i] = NPY_NAN;
        }
    }
    Py_END_ALLOW_THREADS

    /* NOTE: We intentionally keep NaNs in the output profiles. */

    /* ---- Build ROI (if requested) ---- */
    PyArrayObject* roi = NULL;
    if (get_roi) {
        npy_intp rdims[2] = { (npy_intp)(rr1 - rr0), (npy_intp)(cc1 - cc0) };
        roi = (PyArrayObject*)PyArray_SimpleNew(2, rdims, NPY_FLOAT32);
        if (!roi) {
            Py_DECREF(a);
            Py_DECREF(hprof);
            Py_DECREF(vprof);
            Py_XDECREF(mask);
            return NULL;
        }

        float* R = (float*)PyArray_DATA(roi);

        Py_BEGIN_ALLOW_THREADS
        for (int rr = 0; rr < (rr1 - rr0); ++rr) {
            const float* src = A + ((npy_intp)(rr0 + rr)) * ncols + cc0;
            float* dst = R + ((npy_intp)rr) * (cc1 - cc0);
            for (int cc = 0; cc < (cc1 - cc0); ++cc)
                dst[cc] = src[cc];
        }
        Py_END_ALLOW_THREADS
    }

    Py_DECREF(a);
    Py_XDECREF(mask);

    if (get_roi)
        return Py_BuildValue("NNN", vprof, hprof, roi);
    else
        return Py_BuildValue("NN", vprof, hprof);
}
/* ============================================================
  extract_roi_local_f32
  ------------------------------------------------------------
  Extracts a square ROI centered on (x, y) from a float32 2D image.

  ROI definition:
    rows: [x - hl : x + hl)
    cols: [y - hl : y + hl)
  where hl = l / 2

  The ROI is clipped to the image boundaries.

  Requirements:
    - a must be a 2D float32 NumPy array
    - l must be even

  Returns:
    - roi (float32 2D NumPy array)
============================================================ */
static PyObject* py_extract_roi_local_f32(
    PyObject* self, PyObject* args)
{
    PyObject *a_obj = NULL;
    int x, y, l;

    if (!PyArg_ParseTuple(args, "Oiii", &a_obj, &x, &y, &l))
        return NULL;

    /* Convert input array to float32 contiguous */
    PyArrayObject* a = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    if (!a)
        return NULL;

    if (PyArray_NDIM(a) != 2) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_ValueError, "a must be a 2D float32 array");
        return NULL;
    }

    if (l & 1) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_ValueError, "l must be even");
        return NULL;
    }

    npy_intp nrows = PyArray_DIM(a, 0);
    npy_intp ncols = PyArray_DIM(a, 1);
    float* A = (float*)PyArray_DATA(a);

    int hl = l / 2;

    /* Compute ROI bounds */
    int r0 = x - hl;
    int r1 = x + hl;
    int c0 = y - hl;
    int c1 = y + hl;

    int rr0 = (r0 < 0) ? 0 : r0;
    int rr1 = (r1 > (int)nrows) ? (int)nrows : r1;
    int cc0 = (c0 < 0) ? 0 : c0;
    int cc1 = (c1 > (int)ncols) ? (int)ncols : c1;

    npy_intp rdims[2] = {
        rr1 - rr0,
        cc1 - cc0
    };

    PyArrayObject* roi =
        (PyArrayObject*)PyArray_SimpleNew(2, rdims, NPY_FLOAT32);
    if (!roi) {
        Py_DECREF(a);
        return NULL;
    }

    float* R = (float*)PyArray_DATA(roi);

    /* Copy ROI data */
    Py_BEGIN_ALLOW_THREADS
    for (int r = 0; r < rdims[0]; ++r) {
        const float* src =
            A + ((npy_intp)(rr0 + r)) * ncols + cc0;
        float* dst =
            R + ((npy_intp)r) * rdims[1];
        for (int c = 0; c < rdims[1]; ++c) {
            dst[c] = src[c];
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(a);
    return (PyObject*)roi;
}


/* ============================================================
   2) batch_normalize_and_levels
      ----------------------------------------------------------
      Computes 3 normalized/clamped mean levels over 3 pixel
      index lists (left, center, right).

      Equivalent to calling normalize_and_compute_profile_level
      three times, but done in a compact C loop.

      Requirements:
        - a, vmin, vmax: float32 1D
        - left, center, right: int32 1D
        - returns float32[3]
   ============================================================ */

static PyObject* py_batch_normalize_and_levels(
    PyObject* self, PyObject* args)
{
    PyObject *a_obj, *vmin_obj, *vmax_obj;
    PyObject *left_obj, *center_obj, *right_obj;

    if (!PyArg_ParseTuple(
            args, "OOOOOO",
            &a_obj, &vmin_obj, &vmax_obj,
            &left_obj, &center_obj, &right_obj))
        return NULL;

    PyArrayObject *a =
        (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *vmin =
        (PyArrayObject*)PyArray_FROM_OTF(vmin_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *vmax =
        (PyArrayObject*)PyArray_FROM_OTF(vmax_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);

    PyArrayObject *left =
        (PyArrayObject*)PyArray_FROM_OTF(left_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *center =
        (PyArrayObject*)PyArray_FROM_OTF(center_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *right =
        (PyArrayObject*)PyArray_FROM_OTF(right_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);

    if (!a || !vmin || !vmax || !left || !center || !right) {
        Py_XDECREF(a); Py_XDECREF(vmin); Py_XDECREF(vmax);
        Py_XDECREF(left); Py_XDECREF(center); Py_XDECREF(right);
        return NULL;
    }

    if (!ensure_float32_1d(a) ||
        !ensure_float32_1d(vmin) ||
        !ensure_float32_1d(vmax) ||
        !ensure_int32_1d(left) ||
        !ensure_int32_1d(center) ||
        !ensure_int32_1d(right))
    {
        Py_DECREF(a); Py_DECREF(vmin); Py_DECREF(vmax);
        Py_DECREF(left); Py_DECREF(center); Py_DECREF(right);
        PyErr_SetString(PyExc_ValueError, "Invalid array shapes or dtypes");
        return NULL;
    }

    if (PyArray_DIM(a,0) != PyArray_DIM(vmin,0) ||
        PyArray_DIM(a,0) != PyArray_DIM(vmax,0))
    {
        Py_DECREF(a); Py_DECREF(vmin); Py_DECREF(vmax);
        Py_DECREF(left); Py_DECREF(center); Py_DECREF(right);
        PyErr_SetString(PyExc_ValueError, "Size mismatch in a/vmin/vmax");
        return NULL;
    }

    float *A    = (float*)PyArray_DATA(a);
    float *VMIN = (float*)PyArray_DATA(vmin);
    float *VMAX = (float*)PyArray_DATA(vmax);

    int *L = (int*)PyArray_DATA(left);
    int *C = (int*)PyArray_DATA(center);
    int *R = (int*)PyArray_DATA(right);

    npy_intp nL = PyArray_DIM(left,0);
    npy_intp nC = PyArray_DIM(center,0);
    npy_intp nR = PyArray_DIM(right,0);

    npy_intp odims[1] = {3};
    PyArrayObject* out =
        (PyArrayObject*)PyArray_SimpleNew(1, odims, NPY_FLOAT32);
    if (!out) {
        Py_DECREF(a); Py_DECREF(vmin); Py_DECREF(vmax);
        Py_DECREF(left); Py_DECREF(center); Py_DECREF(right);
        return NULL;
    }

    float* O = (float*)PyArray_DATA(out);

    Py_BEGIN_ALLOW_THREADS

    for (int k = 0; k < 3; ++k) {

        int* IDX = (k == 0 ? L : (k == 1 ? C : R));
        npy_intp n = (k == 0 ? nL : (k == 1 ? nC : nR));

        if (n == 0) { O[k] = 0.0f; continue; }

        double acc = 0.0;

        for (npy_intp j = 0; j < n; ++j) {
            int i = IDX[j];
            float denom = VMAX[i] - VMIN[i];
            float x = (denom <= 0.0f) ? 0.0f : (A[i] - VMIN[i]) / denom;
            if (x < 0.0f) x = 0.0f;
            else if (x > 1.0f) x = 1.0f;
            acc += (double) x;
        }

        O[k] = (float)(acc / (double)n);
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(a); Py_DECREF(vmin); Py_DECREF(vmax);
    Py_DECREF(left); Py_DECREF(center); Py_DECREF(right);

    return (PyObject*) out;
}

/* ============================================================
  batch_compute_levels_f32
  ------------------------------------------------------------
  Computes simple mean levels over three pixel index lists
  (left, center, right), without normalization or clamping.

  For each group k:
    output[k] = mean( a[ index_list_k[j] ] )

  Requirements:
    - a: float32 1D NumPy array
    - left, center, right: int32 1D NumPy arrays
  Returns:
    - float32[3] NumPy array: [left_level, center_level, right_level]
============================================================ */
static PyObject* py_batch_compute_levels_f32(
    PyObject* self, PyObject* args)
{
    PyObject *a_obj, *left_obj, *center_obj, *right_obj;
    if (!PyArg_ParseTuple(
            args, "OOOO",
            &a_obj, &left_obj, &center_obj, &right_obj))
        return NULL;

    PyArrayObject *a =
        (PyArrayObject*)PyArray_FROM_OTF(
            a_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *left =
        (PyArrayObject*)PyArray_FROM_OTF(
            left_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *center =
        (PyArrayObject*)PyArray_FROM_OTF(
            center_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *right =
        (PyArrayObject*)PyArray_FROM_OTF(
            right_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);

    if (!a || !left || !center || !right) {
        Py_XDECREF(a);
        Py_XDECREF(left);
        Py_XDECREF(center);
        Py_XDECREF(right);
        return NULL;
    }

    /* Validate array shapes and dtypes */
    if (!ensure_float32_1d(a) ||
        !ensure_int32_1d(left) ||
        !ensure_int32_1d(center) ||
        !ensure_int32_1d(right))
    {
        Py_DECREF(a);
        Py_DECREF(left);
        Py_DECREF(center);
        Py_DECREF(right);
        PyErr_SetString(PyExc_ValueError,
                        "Invalid array shapes or dtypes");
        return NULL;
    }

    float *A = (float*)PyArray_DATA(a);
    int   *L = (int*)PyArray_DATA(left);
    int   *C = (int*)PyArray_DATA(center);
    int   *R = (int*)PyArray_DATA(right);

    npy_intp nL = PyArray_DIM(left, 0);
    npy_intp nC = PyArray_DIM(center, 0);
    npy_intp nR = PyArray_DIM(right, 0);

    npy_intp odims[1] = {3};
    PyArrayObject* out =
        (PyArrayObject*)PyArray_SimpleNew(1, odims, NPY_FLOAT32);
    if (!out) {
        Py_DECREF(a);
        Py_DECREF(left);
        Py_DECREF(center);
        Py_DECREF(right);
        return NULL;
    }

    float* O = (float*)PyArray_DATA(out);

    Py_BEGIN_ALLOW_THREADS
    for (int k = 0; k < 3; ++k) {
        int* IDX = (k == 0 ? L : (k == 1 ? C : R));
        npy_intp n = (k == 0 ? nL : (k == 1 ? nC : nR));

        /* NaN-tolerant mean:
           - ignore NaNs in A[IDX[j]]
           - divide by number of valid samples
           - if no valid sample -> NaN
        */
        double acc = 0.0;
        npy_intp count = 0;

        for (npy_intp j = 0; j < n; ++j) {
            float v = A[IDX[j]];
            if (!is_nan_f32(v)) {
                acc += (double)v;
                count++;
            }
        }

        O[k] = (count > 0) ? (float)(acc / (double)count) : (float)NPY_NAN;
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(a);
    Py_DECREF(left);
    Py_DECREF(center);
    Py_DECREF(right);

    return (PyObject*)out;
}

/* ============================================================
   3) fast_copy_transpose
      ----------------------------------------------------------
      Copies src_2d.T.flatten() into a preallocated 1D float32.

      Requirements:
        - a: float32 2D
        - out: float32 1D of size nrows*ncols

        This respects your viewer orientation rules while
        avoiding Python/NumPy temporary allocations.
   ============================================================ */

static PyObject* py_fast_copy_transpose(PyObject* self, PyObject* args)
{
    PyObject *a_obj, *out_obj;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &out_obj))
        return NULL;

    PyArrayObject* a =
        (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject* out =
        (PyArrayObject*)PyArray_FROM_OTF(out_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);

    if (!a || !out) {
        Py_XDECREF(a);
        Py_XDECREF(out);
        return NULL;
    }

    if (PyArray_NDIM(a) != 2 || PyArray_NDIM(out) != 1) {
        Py_DECREF(a); Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions");
        return NULL;
    }

    npy_intp nrows = PyArray_DIM(a,0);
    npy_intp ncols = PyArray_DIM(a,1);

    if (PyArray_DIM(out,0) != nrows * ncols) {
        Py_DECREF(a); Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError, "Output buffer has incorrect size");
        return NULL;
    }

    float* A = (float*)PyArray_DATA(a);
    float* O = (float*)PyArray_DATA(out);

    /* Column-major traversal of A, which gives A.T.ravel() */
    Py_BEGIN_ALLOW_THREADS

    npy_intp k = 0;
    for (npy_intp c = 0; c < ncols; ++c) {
        const float* col_start = A + c; /* row 0, column c */
        for (npy_intp r = 0; r < nrows; ++r) {
            /* A[r, c] = col_start[r * ncols] */
            O[k++] = col_start[r * ncols];
        }
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(a);
    Py_DECREF(out);

    Py_RETURN_NONE;
}


/* ============================================================
   Module definition
   ============================================================ */

static PyMethodDef Methods[] = {
    {"compute_profiles_local_f32",
        (PyCFunction)py_compute_profiles_local_f32,
        METH_VARARGS | METH_KEYWORDS,
        "Compute H/V profiles and ROI from a float32 2D image"},

    {"batch_normalize_and_levels",
        py_batch_normalize_and_levels, METH_VARARGS,
        "Compute 3 normalized clamped means over pixel index lists"},

    {"fast_copy_transpose",
        py_fast_copy_transpose, METH_VARARGS,
        "Copy a.T.ravel() into an existing float32 1D buffer"},

    {"extract_roi_local_f32",
     py_extract_roi_local_f32, METH_VARARGS,
     "Extract a square ROI centered at (x, y) from a float32 2D image"},

    {"batch_compute_levels_f32",
     py_batch_compute_levels_f32,
     METH_VARARGS,
     "Compute mean levels over left/center/right index lists (no normalization)"},
    
    {"compute_slope_f32",
     py_compute_slope_f32,
     METH_VARARGS,
     "Compute mean(diff(unwrap(x[start:end]))) for float32 array"},
    
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "faster",
    "Fast C kernels for IRCam",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_faster(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
