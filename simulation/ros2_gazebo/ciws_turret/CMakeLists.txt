cmake_minimum_required(VERSION 3.8)
project(ciws_turret)
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf meshes launch worlds config rviz
        DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/spin_slider.py DESTINATION lib/${PROJECT_NAME})
ament_package()
