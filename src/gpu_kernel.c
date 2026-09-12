/* gpu_kernel.c -- kernels compiled at run time (GPU::Kernel).
 *
 * The Ruby side (mrblib/gpu_kernel.rb) traces a block into GLSL source. This
 * file takes that source, runs it through glslangValidator, builds a compute
 * pipeline from the SPIR-V, and dispatches it.
 *
 * Everything here rides on mruby-gpu-narray: the Vulkan context (g_ctx), the
 * descriptor-set layouts, and the buffer helpers all come from its
 * gpu_internal.h. A kernel is one or two input buffers plus one output buffer
 * (the result array, or the per-workgroup partials of a reduction), which is
 * exactly LAYOUT_2BUF / LAYOUT_3BUF, so no new descriptor plumbing is needed.
 *
 * Dispatch goes through mruby-gpu-narray's dispatch_pipeline(), so a traced
 * kernel joins the same deferred batch as the built-in operators: nothing is
 * submitted until a result is read.
 *
 * We shell out to glslangValidator rather than linking glslang, so the link
 * dependency stays libvulkan only -- and the compiler is already required to
 * build mruby-gpu-narray's own shaders.
 */
#include "gpu_internal.h"
#include <mruby/error.h>
#include <unistd.h>

/* Matches GPU::Kernel::REDUCE in mrblib/gpu_kernel.rb. */
enum { REDUCE_NONE = 0, REDUCE_SUM = 1, REDUCE_MIN = 2, REDUCE_MAX = 3 };

typedef struct {
  VkPipeline pipeline;
  int inputs;   /* 1 or 2 input arrays */
  int reduce;   /* REDUCE_NONE for element-wise, else how partials combine */
} GpuKernel;

static void kernel_free(mrb_state *mrb, void *p) {
  GpuKernel *k = (GpuKernel *)p;
  if (!k) return;
  if (k->pipeline != VK_NULL_HANDLE && g_ctx.initialized) {
    vkDestroyPipeline(g_ctx.device, k->pipeline, NULL);
  }
  mrb_free(mrb, k);
}

static const struct mrb_data_type gpu_kernel_type = {
  "GPU::Kernel", kernel_free
};

/* One input -> (a, out) is LAYOUT_2BUF; two inputs -> (a, b, out) is 3BUF. */
static LayoutId kernel_layout(int inputs) {
  return inputs == 2 ? LAYOUT_3BUF : LAYOUT_2BUF;
}

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

/* ---- Vulkan ---- */

static VkPipeline pipeline_from_spv(const char *path, LayoutId lid) {
  size_t spv_size = 0;
  uint8_t *spv = read_file(path, &spv_size);
  if (!spv) return VK_NULL_HANDLE;

  VkShaderModuleCreateInfo sm_info = {
    .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
    .codeSize = spv_size,
    .pCode = (uint32_t *)spv
  };
  VkShaderModule shader;
  if (vkCreateShaderModule(g_ctx.device, &sm_info, NULL, &shader) != VK_SUCCESS) {
    free(spv);
    return VK_NULL_HANDLE;
  }
  free(spv);

  VkComputePipelineCreateInfo cp_info = {
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .stage = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
      .module = shader,
      .pName = "main"
    },
    .layout = g_ctx.pipe_layouts[lid]
  };
  VkPipeline pipeline = VK_NULL_HANDLE;
  if (vkCreateComputePipelines(g_ctx.device, VK_NULL_HANDLE, 1, &cp_info,
                               NULL, &pipeline) != VK_SUCCESS) {
    pipeline = VK_NULL_HANDLE;
  }
  vkDestroyShaderModule(g_ctx.device, shader, NULL);
  return pipeline;
}

/* ---- mruby surface ---- */

/* GPU::Kernel.compile(glsl_source, inputs, reduce_code) -> GPU::Kernel
 *
 * Raises GPU::CompileError carrying glslang's own output, so a bad block reads
 * like a Ruby error rather than a silent failure.
 *
 * Vulkan must already be up. mruby-gpu-narray initializes lazily on its first
 * operation and keeps that entry point private, so the Ruby side touches
 * GPU.info before calling here.
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
  if (!g_ctx.initialized) {
    mrb_raise(mrb, E_RUNTIME_ERROR,
      "no Vulkan device yet -- create a GPU::NArray (or call GPU.info) first");
  }

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

  VkPipeline pipeline = pipeline_from_spv(spv_path, kernel_layout((int)inputs));
  remove(comp_path);
  remove(log_path);
  remove(spv_path);

  if (pipeline == VK_NULL_HANDLE) {
    mrb_exc_raise(mrb, mrb_exc_new_str(mrb, cerr, mrb_str_new_cstr(mrb,
      "SPIR-V compiled but the compute pipeline could not be created")));
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
  GpuBuffer *in[2] = {NULL, NULL};
  for (mrb_int i = 0; i < argc; i++) {
    args[i] = argv[i];
    if (mrb_type(args[i]) != MRB_TT_CDATA || DATA_TYPE(args[i]) != &gpu_buffer_type) {
      mrb_raisef(mrb, E_TYPE_ERROR, "GPU::Kernel#call expects a GPU::NArray, got %s",
                 mrb_obj_classname(mrb, args[i]));
    }
    in[i] = (GpuBuffer *)DATA_PTR(args[i]);
  }
  if (k->inputs == 2 && in[0]->n != in[1]->n) {
    mrb_raisef(mrb, E_ARGUMENT_ERROR, "shape mismatch: %i vs %i",
               (mrb_int)in[0]->n, (mrb_int)in[1]->n);
  }

  uint32_t n = in[0]->n;
  uint32_t groups = (n + 255) / 256;
  LayoutId lid = kernel_layout(k->inputs);
  GpuBuffer *bufs[3] = {in[0], in[1], NULL};   /* inputs, then the output */

  if (k->reduce == REDUCE_NONE) {
    GpuBuffer *out = create_buffer(mrb, n);
    /* Wrapped before dispatching: a Vulkan failure raises, and the GC can only
     * free what it owns. */
    mrb_value result = wrap_buffer(mrb, mrb_obj_class(mrb, args[0]), out);
    bufs[k->inputs] = out;
    dispatch_pipeline(mrb, k->pipeline, lid, bufs, k->inputs + 1, &n, sizeof(n),
                      groups, 1, 1);
    return result;
  }

  /* Reduction: one partial per workgroup, combined here in double precision
   * (the same arrangement as mruby-gpu-narray's #sum). */
  if (n == 0) {
    if (k->reduce == REDUCE_SUM) return mrb_float_value(mrb, 0.0);
    mrb_raise(mrb, E_ARGUMENT_ERROR, "min/max of an empty array");
  }
  GpuBuffer *partial = create_buffer(mrb, groups);
  wrap_buffer(mrb, mrb_obj_class(mrb, args[0]), partial);   /* GC-owned scratch */
  bufs[k->inputs] = partial;
  dispatch_pipeline(mrb, k->pipeline, lid, bufs, k->inputs + 1, &n, sizeof(n),
                    groups, 1, 1);

  /* map_buffer flushes the batch: the partials are its last dispatch. */
  float *pm = map_buffer(mrb, partial);
  double acc = (k->reduce == REDUCE_SUM) ? 0.0 : (double)pm[0];
  for (uint32_t i = 0; i < groups; i++) {
    double v = (double)pm[i];
    switch (k->reduce) {
      case REDUCE_SUM: acc += v; break;
      case REDUCE_MIN: if (v < acc) acc = v; break;
      case REDUCE_MAX: if (v > acc) acc = v; break;
    }
  }
  unmap_buffer(partial);
  return mrb_float_value(mrb, (mrb_float)acc);
}

void mrb_mruby_gpu_kernel_gem_init(mrb_state *mrb) {
  struct RClass *gpu = mrb_define_module(mrb, "GPU");
  mrb_define_class_under(mrb, gpu, "CompileError", mrb_class_get(mrb, "StandardError"));

  struct RClass *kernel = mrb_define_class_under(mrb, gpu, "Kernel", mrb->object_class);
  MRB_SET_INSTANCE_TT(kernel, MRB_TT_CDATA);
  mrb_define_class_method(mrb, kernel, "compile", kernel_s_compile, MRB_ARGS_REQ(3));
  mrb_define_method(mrb, kernel, "call", kernel_call, MRB_ARGS_ANY());
}

void mrb_mruby_gpu_kernel_gem_final(mrb_state *mrb) {
  (void)mrb;
}
