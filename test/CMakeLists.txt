add_executable(eigen_demo eigen_test.cpp)
add_executable(psf_test psf_test.cpp)
add_executable(fmin_search_test fmin_search_test.cpp)

TARGET_LINK_LIBRARIES(eigen_demo ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(psf_test ${OpenCV_LIBS} ${itpp_lib} ${fftw_lib})
TARGET_LINK_LIBRARIES(fmin_search_test ${OpenCV_LIBS} ${itpp_lib} ${fftw_lib})
