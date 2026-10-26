cmake_minimum_required(VERSION 3.8)
project(ciws_turret)
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf meshes launch worlds config rviz
        DESTINATION share/${PROJECT_NAME})
ament_package()
