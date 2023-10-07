---
title: "Turtlebot3 사용을 위한 ROS Package 생성"
date: 2023-10-04
categories: ROS Robot operation system turtlebot turtlebot3 AMR AGV Package catkin cmake
---

# Turtlebot3 사용을 위한 ROS Package 생성

- catkin(캐킨)은 ROS의 빌드 시스템
- CMake(Cross Platform Make)를 기본적으로 이용하여 패키지 폴더에 CMakeList.txt라는 파일에 빌드 환경을 기술해야함
- ROS에서는 CMake를 ROS에 맞게 수정해 특화된 캐킨 빌드 시스템을 만들었으며, ROS 관련 빌드, 패키지 관리, 패키지 간 의존관계 등을 편리하게 사용할 수 있게 됨
- 경로는 catkin_ws/src/turtlebot 에서 실행

## 패키지 생성
- 패키지 생성을 위해 터미널을 열고 경로를 이동한다

```commandline
catkin_create_pkg 패키지이름 std_msgs rospy roscpp
```

- 의존성 패키지는 해당 패키지에서 사용할 의존성을 기술
- std_msgs: 주고 받을 메시지 타입,rospy: python 사용, roscpp: c++ 사용
- 의존성 패키지는 여러 개를 동시선언할 수 있고, 추후에 pakage.xml에서 추가/변경할 수 있음
- 사용자가 패키지 작성 시, 캐킨 빌드 시스템에 필요한 CMakeList.txt, pakage.xml, 관련 폴더 생성
- 패키지 이름에는 공백이 있어선 안되며 소문자를 사용하고 언더바(_)를 사용해 단어를 붙임

## 패키지 설정 파일(package.xml) 수정

- package.xml는 패키지의 이름, 저작사, 라이센스, 의존성 패키지 등을 기술하고 있는 파일로 ROS의 필수 설정 파일의 하나

### 기본 구조

- ?xml: 문서 문법을 정의하는 문구. xml 버전을 나타냄
- package: 해당 태그로 감싼 부분이 ROS 패키지 설정 부분

### 패키지 정보

- name: 패키지의 이름. 패키지 생성 시 입력한 이름이 적용되며 사용자의 임의변경이 가능
- version: 패키지 버전으로 자유로운 지정이 가능
- description: 패키지에 대한 설명으로 2~3문장으로 입력
- maintainer: 패키지 관리자의 이름과 메일 주소(태그의 옵션 email을 이용)를 입력
- license: 라이센스를 기재(e.g. GPL, BSD, ASL)
- url: 패키지를 설명하는 웹페이지, 버그 관리, 저장소 등의 주소
- author: 패키지 개발에 참여한 개발자의 이름과 이메일 주소를 적음. 여러명의 개발자의 경우 바로 다음줄에 해당 태그를 추가하며 입력

### 의존 패키지(Dependency)

- buildtool_depend: 빌드 시스템의 의존성이며, Catkin 빌드 시스템을 이용한다면 catkin을 입력
- build_depend: 패키지를 빌드할 때 의존하는 패키지 이름을 입력
- run_depend: 패키지를 실행할 때 의존하는 패키지 이름을 입력
- test_depend: 패키지를 테스트할 때 의존하는 패키지 이름을 입력

### 메타패키지(Metapackage)

- export: ROS에서 명시하지 않은 태그명을 사용할 때 쓰임
- metapackage: export 태그 안에서 사용하는 공식적인 태그 중 하나로 현재 패키지가 메타패키지일 경우 선언

## 빌드 설정 파일(CMakeList.txt) 수정

- CMakeList.txt는 빌드 환경을 기술하고 있는 파일로 실행 파일 생성과 의존성 패키지 우선 빌드, 링크 생성 등을 설정

```text
# 운영체제에 설치된 cmake의 최소 요구 버전
cmake_minimum_required(VERSION 3.0.2)

# 패키지의 이름으로, package.xml에서 입력한 패키지 이름을 그대로 사용
project(test_pkg)

# 캐킨 빌드할 시 요구되는 구성 요소 패키지. 사용자가 만든 패키지가 의존하는 다른 패키지를 먼저 설치하는 옵션
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

# ROS 이외의 패키지를 사용하는 예: Boost를 사용할 때 system 패키지를 설치하도록 함
find_package(Boost REQUIRED COMPONENTS system)

# 파이썬을 이용하기 위해 rospy를 사용할 때 설정하는 옵션. 파이썬 설치 프로세스인 setup.py를 부르는 역할
catkin_python_setup()

# 메시지 파일을 추가
# FILES: 현재 패키지 폴더의 msg 폴더 안의 .msg 파일들을 참조해 헤더 파일(.h)를 자동으로 생성한다는 의미
# 만약 새 메시지를 만든다면 msg 폴더를 만든 뒤 그 안에 있는 메시지 파일 이름을 입력함. 여기에서는 MyMessage1.msg 등이 그 예.
add_message_files(
  FILES 
  MyMessage1.msg
  MyMessage2.msg
)

# 사용하는 서비스 파일을 추가. 방식은 메시지 파일과 같으며, 사용하려면 srv 폴더를 만든 뒤 해당 파일 이름을 입력해둬야 한다.
add_service_files(
  FILES
  MyService.srv
)

# 사용하는 서비스 파일을 추가. 방식은 메시지, 서비스 파일과 같다.
add_action_files(
  FILES
  Action1.action
  Action2.action
)

# 의존하는 메시지를 설정
# DEPENDENCIES: 아래에 해당하는 메시지 패키지를 사용한다는 의미
# std_msgs, sensor_msgs가 그 예시
generate_messages(
  std_msgs 
  sensor_msgs
)

# 캐킨 빌드 옵션
## INCLUDE: 뒤에 설정한 패키지 내부 폴더인 include의 헤더 파일을 사용함
## LIBRARIES: 뒤에 설정한 패키지의 라이브러리를 사용함
## CATKIN_DEPENDS: 의존하는 패키지 지정
## DEPENDS: 시스템 의존 패키지
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES test_pkg
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)

# include 폴더 지정
include_directories(
  ${catkin_INCLUDE_DIRS} # 각 패키지 내의 include 폴더를 의미. 이 안의 헤더파일을 이용할 것. 
  # 사용자가 추가할 때는 이 밑의 공간 이용
)

# 빌드 후 생성할 라이브러리. C++을 사용할 경우!
add_library(${PROJECT_NAME}
  src/${PROJECT_NAME}/test_pkg.cpp
)

# 해당 라이브러리 및 실행파일을 빌드하기 전, 생성해야 할 의존성이 있는 메시지와 dynamic reconfigure이 있다면 우선으로 수행하도록 함
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# 빌드 후 생성할 실행파일에 대한 옵션 지정
## `__실행 파일 이름__` `__참조할 파일__` 순서대로 기재
## 복수 개의 참조 .cpp 파일이 있을 경우 한 괄호 뒤에 연속적으로 기재
## 생성할 실행파일이 2개 이상일 경우 add_executable 항목을 추가함
add_executable(${PROJECT_NAME}_node src/test_pkg_node.cpp)

# 지정 실행 파일을 생성하기 전, 링크해야 하는 라이브러리와 실행파일을 링크함
target_link_libraries(${PROJECT_NAME}_node
  ${catkin_LIBRARIES}
)
```

## 빌드 전 처리
- package.xml 파일 속 다음 부분을 추가

```xml
  <build_depend>message_generation</build_depend>
  <exec_depend>message_runtime</exec_depend>
```

- CMakeList.txt 파일 속 다음 부분을 주석을 풀거나 수정

```text
# message_generation을 추가한다.
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)

# add_message_files의 주석을 풀고 수정한다.
add_message_files(
  FILES
  test_msg.msg
)

# generate_messages의 주석을 푼다.
generate_messages(
  DEPENDENCIES
  std_msgs
)

# LIBRARIES, CATKIN_DEPENDS의 주석을 풀고, message_runtime을 추가한다.
catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES test_pkg
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime
#  DEPENDS system_lib
)

# 주석을 풀고 스크립트 이름을 입력한다. talker_py.py, listener.py를 파이썬으로 사용한다는 이야기
catkin_install_python(PROGRAMS
  src/talker_py.py
  src/listener_py.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
```

- 수정 사항을 포함해 패키지를 빌드

```commandline
catkin_make
```