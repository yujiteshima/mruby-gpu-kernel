MRuby::Gem::Specification.new('mruby-gpu-kernel') do |spec|
  spec.license = 'MIT'
  spec.authors = 'Yuji Teshima'
  spec.version = '0.1.0'
  spec.summary = 'Trace a Ruby block into a Vulkan compute shader for mruby-gpu-narray'

  # Resolved from mgem-list when no source is given (lib/mruby/gem.rb:
  # default_gem_params falls back to { mgem: <name> }).
  spec.add_dependency 'mruby-gpu-narray'

  spec.linker.libraries << 'vulkan'

  # This gem compiles against mruby-gpu-narray's gpu_internal.h -- it uses the
  # shared Vulkan context (g_ctx), the descriptor-set layouts, and the buffer
  # helpers, all of which that header exports. The base gem can arrive from
  # mgem-list, from GitHub, or from a local path, and dependencies are cloned
  # after this file is evaluated, so we add every plausible location of its
  # src/. A -I pointing at a directory that does not exist is ignored, which
  # makes listing candidates safe.
  candidates = []
  base = build.gems['mruby-gpu-narray']
  candidates << base.dir if base
  candidates << ENV['MRUBY_GPU_NARRAY_DIR'] if ENV['MRUBY_GPU_NARRAY_DIR']
  candidates << "#{build.gem_clone_dir}/mruby-gpu-narray"
  candidates << File.expand_path('../mruby-gpu-narray', File.dirname(File.expand_path(__FILE__)))
  candidates.uniq.each { |d| spec.cc.include_paths << File.join(d, 'src') }
end
