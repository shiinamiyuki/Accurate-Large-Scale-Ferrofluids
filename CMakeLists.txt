cmake_minimum_required(VERSION 2.6)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake
  CACHE STRING "Vcpkg toolchain file")
project(ferror-fluids)



# find_package(CUDAToolkit REQUIRED)
option(LIBIGL_WITH_COMISO            "Use CoMiso"                   OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"                   OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"                   ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                     ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    ON)
option(LIBIGL_WITH_PNG               "Use PNG"                      ON)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"                   OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 ON)
option(LIBIGL_WITH_PREDICATES        "Use exact predicates"         ON)
option(LIBIGL_WITH_XML               "Use XML"                      OFF)
option(LIBIGL_WITH_PYTHON            "Use Python"                   OFF)
option(LIBIGL_USE_STATIC_LIBRARY "Use libigl as static library"     OFF)

# SET(CUDA_SEPARABLE_COMPILATION ON)
# find_package(CUDAToolkit)
# include (CheckLanguage)
# check_language(CUDA)
# enable_language(CUDA)

# # FIXME
# include_directories (${CUDAToolkit_INCLUDE_DIRS})  # for regular c++ compiles

# # http://www.ssl.berkeley.edu/~jimm/grizzly_docs/SSL/opt/intel/cc/9.0/lib/locale/en_US/mcpcom.msg
# set (CUDA_DIAG_FLAGS "")
# #set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} -Xptxas --warn-on-double-precision-use")
# set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=partial_override")
# set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=virtual_function_decl_hidden")
# set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=integer_sign_change")
# set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=declared_but_not_referenced")
# set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=field_without_dll_interface")
# # WAR invalid warnings about this with "if constexpr"
# set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
# set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} -Xcudafe --diag_suppress=implicit_return_from_non_void_function")
# set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} --expt-relaxed-constexpr")
# set (CUDA_DIAG_FLAGS "${CUDA_DIAG_FLAGS} --extended-lambda")
# set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_DIAG_FLAGS}")

# # Willie hears yeh..
# set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xnvlink -suppress-stack-size-warning")
# set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17 -rdc=true --keep --use_fast_math -maxrregcount 128")

# macro (set_target_CUDA_props target)
#         set_target_properties(${target} 
#         PROPERTIES  CUDA_ARCHITECTURES 72
#                     CUDA_SEPARABLE_COMPILATION  ON )
# endmacro()

# add_library(sim_core src/simulation.h src/simulation.cu)
# add_executable(sim src/main.cpp)
# target_link_libraries(sim_core glm CUDA::toolkit CUDA::cudart CUDA::cudart_static CUDA::cuda_driver)
# target_link_libraries(sim sim_core igl::opengl igl::opengl_glfw igl::core)
# set_target_CUDA_props(sim_core)
# set_target_CUDA_props(sim)
# # target_include_directories(sim )
find_package(glm CONFIG REQUIRED)
find_package(libigl CONFIG REQUIRED)
find_package(tbb CONFIG REQUIRED)
add_executable(sim src/main.cpp src/simulation.h src/simulation.cpp src/reconstruction.cpp)
target_link_libraries(sim glm igl::opengl igl::opengl_glfw igl::common TBB::tbb)