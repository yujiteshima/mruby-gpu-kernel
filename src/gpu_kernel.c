/* gpu_kernel.c -- kernels compiled at run time (GPU::Kernel).
 *
 * The Ruby side (mrblib/gpu_kernel.rb) traces a block into GLSL source. This
 * file takes that source, runs it through glslangValidator, builds a compute
 * pipeline from the SPIR-V, and dispatches it.
 *
 * Everything on the GPU goes through mruby-gpu-runtime's C API
 * (gpu_runtime.h): the pipeline is created from the SPIR-V by the runtime, the
 * arrays that Ruby passes in are read as its buffers, and the result is a
 * buffer it creates. A kernel is one or two input buffers plus one output
 * buffer (the result array, or the per-workgroup partials of a reduction), so
 * its layout is two or three storage buffers and four bytes of push constants.
 * This file does not call Vulkan itself.
 *
 * Dispatch goes through gpu_rt_dispatch(), so a traced kernel joins the same
 * deferred batch as mruby-gpu-narray's built-in operators: nothing is
 * submitted until a result is read.
 *
 * We shell out to glslangValidator rather than linking glslang, so the link
 * dependency stays libvulkan only -- and the compiler is already required to
 * build mruby-gpu-narray's own shaders.
 */
#include <gpu_runtime.h>
#include <mruby/class.h>
#include <mruby/data.h>
#include <mruby/error.h>
#include <mruby/string.h>
#include <mruby/variable.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Matches GPU::Kernel::REDUCE in mrblib/gpu_kernel.rb. */
enum { REDUCE_NONE = 0, REDUCE_SUM = 1, REDUCE_MIN = 2, REDUCE_MAX = 3 };

typedef struct {
  gpu_rt_pipeline *pipeline;
  int inputs;   /* 1 or 2 input arrays */
  int reduce;   /* REDUCE_NONE for element-wise, else how partials combine */
} GpuKernel;

static void kernel_free(mrb_state *mrb, void *p) {
  GpuKernel *k = (GpuKernel *)p;
  if (!k) return;
  /* Safe from the GC: the runtime keeps the pipeline until the dispatches
   * that are still only recorded have run. */
  gpu_rt_pipeline_release(mrb, k->pipeline);
  mrb_free(mrb, k);
}

static const struct mrb_data_type gpu_kernel_type = {
  "GPU::Kernel", kernel_free
};

/* ---- file helpers ---- */

static uint8_t *read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (len < 0) { fclose(f); return NULL; }
  uint8_t *buf = (uint8_t *)malloc((size_t)len + 1);
  if (buf) {
    size_t got = fread(buf, 1, (size_t)len, f);
    buf[got] = '\0';
    if (size) *size = got;
  }
  fclose(f);
  return buf;
}

/* ---- mruby surface ---- */

/* GPU::Kernel.compile(glsl_source, inputs, reduce_code) -> GPU::Kernel
 *
 * Raises GPU::CompileError carrying glslang's own output, so a bad block reads
 * like a Ruby error rather than a silent failure.
 *
 * Starts the device if nothing has used the GPU yet.
 */
static mrb_value kernel_s_compile(mrb_state *mrb, mrb_value self) {
  const char *src;
  mrb_int src_len, inputs, reduce;
  mrb_get_args(mrb, "sii", &src, &src_len, &inputs, &reduce);

  struct RClass *gpu = mrb_module_get(mrb, "GPU");
  struct RClass *cerr = mrb_class_get_under(mrb, gpu, "CompileError");

  if (inputs != 1 && inputs != 2) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "a kernel takes 1 or 2 inputs, got %i", inputs);
  }
  if (reduce < REDUCE_NONE || reduce > REDUCE_MAX) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "unknown reduction code %i", reduce);
  }
  gpu_rt_context *ctx = gpu_rt_context_get(mrb);

  const char *tmpdir = getenv("TMPDIR");
  if (!tmpdir || !*tmpdir) tmpdir = "/tmp";

  static unsigned long seq = 0;
  unsigned long id = seq++;
  char comp_path[1024], spv_path[1024], log_path[1024], cmd[4096];
  snprintf(comp_path, sizeof(comp_path), "%s/mruby-gpu-kernel-%ld-%lu.comp",
           tmpdir, (long)getpid(), id);
  snprintf(spv_path, sizeof(spv_path), "%s/mruby-gpu-kernel-%ld-%lu.spv",
           tmpdir, (long)getpid(), id);
  snprintf(log_path, sizeof(log_path), "%s/mruby-gpu-kernel-%ld-%lu.log",
           tmpdir, (long)getpid(), id);

  FILE *f = fopen(comp_path, "wb");
  if (!f) mrb_raisef(mrb, E_RUNTIME_ERROR, "cannot write %s", comp_path);
  fwrite(src, 1, (size_t)src_len, f);
  fclose(f);

  const char *glslang = getenv("GLSLANG");
  if (!glslang || !*glslang) glslang = "glslangValidator";

  snprintf(cmd, sizeof(cmd), "\"%s\" -V \"%s\" -o \"%s\" > \"%s\" 2>&1",
           glslang, comp_path, spv_path, log_path);
  int rc = system(cmd);

  if (rc != 0) {
    uint8_t *log = read_file(log_path, NULL);
    mrb_value msg = mrb_str_new_cstr(mrb, log ? (const char *)log
      : "glslangValidator failed (is it on PATH? override with GLSLANG=)");
    free(log);
    remove(comp_path);
    remove(log_path);
    remove(spv_path);
    mrb_exc_raise(mrb, mrb_exc_new_str(mrb, cerr, msg));
  }

  size_t spv_size = 0;
  uint8_t *spv = read_file(spv_path, &spv_size);
  remove(comp_path);
  remove(log_path);
  remove(spv_path);
  if (!spv) {
    mrb_exc_raise(mrb, mrb_exc_new_str(mrb, cerr, mrb_str_new_cstr(mrb,
      "glslangValidator reported success but its SPIR-V output could not be read")));
  }

  /* (a, out) or (a, b, out), and { uint n; } as push constants. The error
   * struct makes the runtime report a shader the device refuses instead of
   * raising, so that it becomes a GPU::CompileError like any other bad block. */
  gpu_rt_layout layout = {
    .storage_buffers = (uint32_t)inputs + 1,
    .push_constant_size = sizeof(uint32_t)
  };
  gpu_rt_error err;
  gpu_rt_pipeline *pipeline = gpu_rt_pipeline_create(mrb, ctx, spv, spv_size, &layout,
                                                     "kernel", &err);
  free(spv);
  if (!pipeline) {
    char msg[256];
    snprintf(msg, sizeof(msg),
      "SPIR-V compiled but the compute pipeline could not be created: "
      "%s failed: %s (VkResult %d)", err.operation, err.reason, (int)err.vk_result);
    mrb_exc_raise(mrb, mrb_exc_new_str(mrb, cerr, mrb_str_new_cstr(mrb, msg)));
  }

  GpuKernel *k = (GpuKernel *)mrb_malloc(mrb, sizeof(GpuKernel));
  k->pipeline = pipeline;
  k->inputs = (int)inputs;
  k->reduce = (int)reduce;

  struct RObject *obj = (struct RObject *)mrb_obj_alloc(mrb, MRB_TT_CDATA,
                                                        mrb_class_ptr(self));
  mrb_value kernel = mrb_obj_value(obj);
  DATA_PTR(kernel) = k;
  DATA_TYPE(kernel) = &gpu_kernel_type;
  mrb_iv_set(mrb, kernel, mrb_intern_lit(mrb, "@glsl"),
             mrb_str_new(mrb, src, src_len));
  return kernel;
}

/* kernel.call(a [, b]) -> GPU::NArray for an element-wise kernel, or the
 * reduced Float for a reduction. */
static mrb_value kernel_call(mrb_state *mrb, mrb_value self) {
  const mrb_value *argv;
  mrb_int argc;
  mrb_get_args(mrb, "*", &argv, &argc);

  GpuKernel *k = (GpuKernel *)DATA_GET_PTR(mrb, self, &gpu_kernel_type, GpuKernel);

  if (argc != k->inputs) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR,
      "this kernel takes %i array(s) (its block has that many parameters), got %i",
      (mrb_int)k->inputs, argc);
  }

  /* Copy out of the VM stack before allocating anything. */
  mrb_value args[2] = {mrb_nil_value(), mrb_nil_value()};
  gpu_rt_buffer *in[2] = {NULL, NULL};
  uint32_t len[2] = {0, 0};   /* float32 elements */
  for (mrb_int i = 0; i < argc; i++) {
    args[i] = argv[i];
    in[i] = gpu_rt_buffer_check(mrb, args[i]);
    if (!in[i]) {
      mrb_raisef(mrb, E_TYPE_ERROR, "GPU::Kernel#call expects a GPU::NArray, got %s",
                 mrb_obj_classname(mrb, args[i]));
    }
    len[i] = (uint32_t)(gpu_rt_buffer_size(mrb, in[i]) / sizeof(float));
  }
  if (k->inputs == 2 && len[0] != len[1]) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "shape mismatch: %i vs %i",
               (mrb_int)len[0], (mrb_int)len[1]);
  }

  gpu_rt_context *ctx = gpu_rt_context_get(mrb);
  uint32_t n = len[0];
  uint32_t groups = (n + 255) / 256;
  gpu_rt_buffer *bufs[3] = {in[0], in[1], NULL};   /* inputs, then the output */
  gpu_rt_buffer_desc out_desc = { .usage = GPU_RT_BUFFER_STORAGE };
  gpu_rt_dispatch_desc dispatch = {
    .pipeline = k->pipeline,
    .buffers = bufs, .buffer_count = (uint32_t)k->inputs + 1,
    .push_constants = &n, .push_constant_size = sizeof(n),
    .groups = { groups },
    .label = "kernel",
  };

  if (k->reduce == REDUCE_NONE) {
    out_desc.size = (size_t)n * sizeof(float);
    gpu_rt_buffer *out = gpu_rt_buffer_create(mrb, ctx, &out_desc);
    /* Wrapped before dispatching: a failed dispatch raises, and the GC can
     * only free what it owns. */
    mrb_value result = gpu_rt_buffer_wrap(mrb, mrb_obj_class(mrb, args[0]), out);
    bufs[k->inputs] = out;
    gpu_rt_dispatch(mrb, &dispatch);
    return result;
  }

  /* Reduction: one partial per workgroup, combined here in double precision
   * (the same arrangement as mruby-gpu-narray's #sum). */
  if (n == 0) {
    if (k->reduce == REDUCE_SUM) return mrb_float_value(mrb, 0.0);
    mrb_raise(mrb, E_ARGUMENT_ERROR, "min/max of an empty array");
  }
  out_desc.size = (size_t)groups * sizeof(float);
  gpu_rt_buffer *partial = gpu_rt_buffer_create(mrb, ctx, &out_desc);
  gpu_rt_buffer_wrap(mrb, mrb_obj_class(mrb, args[0]), partial);   /* GC-owned scratch */
  bufs[k->inputs] = partial;
  gpu_rt_dispatch(mrb, &dispatch);

  /* Mapping runs the batch: the partials are its last dispatch. */
  float *pm = (float *)gpu_rt_buffer_map(mrb, partial);
  double acc = (k->reduce == REDUCE_SUM) ? 0.0 : (double)pm[0];
  for (uint32_t i = 0; i < groups; i++) {
    double v = (double)pm[i];
    switch (k->reduce) {
      case REDUCE_SUM: acc += v; break;
      case REDUCE_MIN: if (v < acc) acc = v; break;
      case REDUCE_MAX: if (v > acc) acc = v; break;
    }
  }
  gpu_rt_buffer_unmap(mrb, partial);
  return mrb_float_value(mrb, (mrb_float)acc);
}

void mrb_mruby_gpu_kernel_gem_init(mrb_state *mrb) {
  /* Written against version 0.1 of the runtime's C API. */
  gpu_rt_require(mrb, 0, 1);

  struct RClass *gpu = mrb_define_module(mrb, "GPU");
  /* mruby-gpu-runtime defines GPU::CompileError under GPU::Error once it has
   * the GPU::Error hierarchy (mruby-gpu-runtime#5). Defining it again with
   * another superclass would raise, so it is defined here only when the
   * runtime has not: this file works with a runtime from before and after. */
  if (!mrb_class_defined_under(mrb, gpu, "CompileError")) {
    mrb_define_class_under(mrb, gpu, "CompileError", mrb_class_get(mrb, "StandardError"));
  }

  struct RClass *kernel = mrb_define_class_under(mrb, gpu, "Kernel", mrb->object_class);
  MRB_SET_INSTANCE_TT(kernel, MRB_TT_CDATA);
  mrb_define_class_method(mrb, kernel, "compile", kernel_s_compile, MRB_ARGS_REQ(3));
  mrb_define_method(mrb, kernel, "call", kernel_call, MRB_ARGS_ANY());
}

void mrb_mruby_gpu_kernel_gem_final(mrb_state *mrb) {
  (void)mrb;
}
