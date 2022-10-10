# --------------------------------------------------------------------------------------------------
# Variables definition
# --------------------------------------------------------------------------------------------------

set(CHALLENGE_LIBNAME challenge)

file(GLOB_RECURSE CHALLENGE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

if(KL_TARGET_OS STREQUAL "linux")
  set(TFLITE_PATH ${VENDORS_DIR}/tflite/lib/linux/x86_64/libtensorflowlite.a)
  set(LIBJPEG_PATH ${VENDORS_DIR}/opencv2/lib/linux/x86_64/liblibjpeg.a)
  set(LIBPNG_PATH ${VENDORS_DIR}/opencv2/lib/linux/x86_64/liblibpng.a)
  set(LIBTIFF_PATH ${VENDORS_DIR}/opencv2/lib/linux/x86_64/liblibtiff.a)
  set(OPENCV_CORE_PATH ${VENDORS_DIR}/opencv2/lib/linux/x86_64/libopencv_core.a)
  set(OPENCV_IMGPROC_PATH ${VENDORS_DIR}/opencv2/lib/linux/x86_64/libopencv_imgproc.a)
  set(OPENCV_HIGHGUI_PATH ${VENDORS_DIR}/opencv2/lib/linux/x86_64/libopencv_highgui.a)
  set(ZLIB_PATH ${VENDORS_DIR}/opencv2/lib/linux/x86_64/libzlib.a)
elseif(KL_TARGET_OS STREQUAL "macos")
  set(TFLITE_PATH ${VENDORS_DIR}/tflite/lib/macos/x86_64/libtensorflowlite.a)
  set(LIBJPEG_PATH ${VENDORS_DIR}/opencv2/lib/macos/x86_64/liblibjpeg.a)
  set(LIBPNG_PATH ${VENDORS_DIR}/opencv2/lib/macos/x86_64/liblibpng.a)
  set(LIBTIFF_PATH ${VENDORS_DIR}/opencv2/lib/macos/x86_64/liblibtiff.a)
  # OpenCV libs
  set(OPENCV_CORE_PATH ${VENDORS_DIR}/opencv2/lib/macos/x86_64/libopencv_core.a)
  set(OPENCV_IMGPROC_PATH ${VENDORS_DIR}/opencv2/lib/macos/x86_64/libopencv_imgproc.a)
  set(OPENCV_HIGHGUI_PATH ${VENDORS_DIR}/opencv2/lib/macos/x86_64/libopencv_highgui.a)
  set(ZLIB_PATH ${VENDORS_DIR}/opencv2/lib/macos/x86_64/libzlib.a)
endif()

# --------------------------------------------------------------------------------------------------
# Set dependencies
# --------------------------------------------------------------------------------------------------

if(KL_TARGET_OS STREQUAL "macos")
  add_library(${CHALLENGE_LIBNAME}
    STATIC ${CHALLENGE_SRCS}
  )
else()
  add_library(${CHALLENGE_LIBNAME}
    STATIC ${CHALLENGE_SRCS}
  )
endif()

# --------------------------------------------------------------------------------------------------
# Build rules
# --------------------------------------------------------------------------------------------------

# Some files are generated in the build folder to not pollute the repo
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(KL_TARGET_OS STREQUAL "linux")
  add_custom_command(TARGET ${CHALLENGE_LIBNAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND}
      -DTFLITE_PATH=${TFLITE_PATH}
      -DLIBJPEG_PATH=${LIBJPEG_PATH}
      -DLIBPNG_PATH=${LIBPNG_PATH}
      -DLIBTIFF_PATH=${LIBTIFF_PATH}
      -DOPENCV_CORE_PATH=${OPENCV_CORE_PATH}
      -DOPENCV_HIGHGUI_PATH=${OPENCV_HIGHGUI_PATH}
      -DOPENCV_IMGPROC_PATH=${OPENCV_IMGPROC_PATH}
      -DZLIB_PATH=${ZLIB_PATH}
      -P ${MAKE_DIR}/generate_ar_input_file.cmake
    COMMAND ar -M < ${CMAKE_CURRENT_BINARY_DIR}/ar_input.txt
    COMMAND ranlib libchallenge.a.tmp
    COMMAND mv libchallenge.a.tmp libchallenge.a
    # COMMAND ${CMAKE_STRIP} --strip-unneeded libchallenge.a
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Joining libs into one"
  )

elseif(KL_TARGET_OS STREQUAL "macos")
  add_custom_command(TARGET ${CHALLENGE_LIBNAME} POST_BUILD
    COMMAND libtool -static -o libchallenge.a
      libchallenge.a
      ${TFLITE_PATH}
      ${LIBJPEG_PATH}
      ${LIBPNG_PATH}
      ${LIBTIFF_PATH}
      ${OPENCV_CORE_PATH}
      ${OPENCV_IMGPROC_PATH}
      ${OPENCV_HIGHGUI_PATH}
      ${ZLIB_PATH}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Joining libs into one"
  )
endif()
