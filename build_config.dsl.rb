# Build config for the dsl-control-flow branch (development only).
#   cd ~/dev/personal/mruby-dsl && MRUBY_CONFIG=~/dev/personal/mruby-gpu-kernel-dsl/build_config.dsl.rb rake -j8
MRuby::Build.new do |conf|
  toolchain :clang
  conf.gembox 'default'
  conf.gem File.expand_path('~/dev/personal/mruby-gpu-narray-base')   # listed first so mgem-list is not consulted
  conf.gem File.expand_path('~/dev/personal/mruby-gpu-kernel-dsl')
  conf.cc.include_paths     << '/opt/homebrew/include'
  conf.linker.library_paths << '/opt/homebrew/lib'
end
