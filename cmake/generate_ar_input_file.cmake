file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ar_input.txt "
  CREATE libchallenge.a.tmp
  ADDLIB ${TFLITE_PATH}
  ADDLIB ${LIBJPEG_PATH}
  ADDLIB ${LIBPNG_PATH}
  ADDLIB ${LIBTIFF_PATH}
  ADDLIB ${OPENCV_CORE_PATH}
  ADDLIB ${OPENCV_IMGPROC_PATH}
  ADDLIB ${OPENCV_HIGHGUI_PATH}
  ADDLIB ${ZLIB_PATH}
  ADDLIB libchallenge.a
  SAVE
  END
")
