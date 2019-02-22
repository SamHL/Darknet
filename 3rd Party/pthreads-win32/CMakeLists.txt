#=======================================================
# PTHREADS WIN32
#=======================================================
cmake_minimum_required(VERSION 3.9)
PROJECT(pthreads)
# This cmake file exports variables to facilitate building
# pthreads as part of another project. 
# 
# EXPORTED VARIABLES:
# 		PTHREAD_API
#		PTHREADS_COMPATIBILITY_VERSION
# 		PTHREADS_LIBRARY
#=======================================================
# PTHREADS COMPILE DEFINITIONS
#=======================================================
set(PTHREADS_COMPILE_DEFINITIONS "")

	# Size of void
if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
	list(APPEND PTHREADS_COMPILE_DEFINITIONS -DPTW32_ARCH="x64")
else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
	list(APPEND PTHREADS_COMPILE_DEFINITIONS -DPTW32_ARCH="x86")
endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )

	# Check for compiler 
if(MSVC)
	list(APPEND PTHREADS_COMPILE_DEFINITIONS /DPTW32_RC_MSC)
	set(PTHREADS_COMPILER V)
elseif(CMAKE_COMPILER_IS_GNUCXX)
	set(PTHREADS_COMPILER G)
else()
	message("WARNING: pthreads-win32 doesn't recognize your compiler!")
endif()

	# Exception scheme (add definitions)
if(PTHREADS_BUILD_CPP)
	set(PTHREADS_EXCEPTION_SCHEME CE)
	list(APPEND PTHREADS_COMPILE_DEFINITIONS /__CLEANUP_CXX)
	message(STATUS "Building pthreads-win32 with C++ exception handling.")
elseif(PTHREADS_BUILD_SEH)
	set(PTHREADS_EXCEPTION_SCHEME SE)
	list(APPEND PTHREADS_COMPILE_DEFINITIONS /__CLEANUP_SEH)
	message(STATUS "Building pthreads-win32 with Microsoft SEH exception handling.")
else()
	# defaults to C - setjmp/longjmp
	set(PTHREADS_EXCEPTION_SCHEME C)
	message(STATUS "Building pthreads-win32 with C setjmp/longjmp (default/recommended exception scheme).")
endif()

list(APPEND PTHREADS_COMPILE_DEFINITIONS /DHAVE_PTW32_CONFIG_H)

foreach(def ${PTHREADS_COMPILE_DEFINITIONS})
	add_definitions(${def})
endforeach()

#=======================================================
# PTHREADS COMPATIBILITY VERSIONING
#=======================================================
set(PTHREADS_COMPATIBILITY_VERSION 2)
set(PTHREADS_LIBRARY "pthreads${PTHREADS_COMPILER}${PTHREADS_EXCEPTION_SCHEME}${PTHREADS_COMPATIBILITY_VERSION}" PARENT_SCOPE)

#=======================================================
# GLOB FILES 
#=======================================================
set(PTHREAD_API
	${PROJECT_SOURCE_DIR}/pthread.h
	${PROJECT_SOURCE_DIR}/sched.h
	${PROJECT_SOURCE_DIR}/semaphore.h
	PARENT_SCOPE)

set(PTHREAD_SOURCES
	attr.c
	barrier.c
	cancel.c
	cleanup.c
	condvar.c
	create.c
	dll.c
	autostatic.c
	errno.c
	exit.c
	fork.c
	global.c
	misc.c
	mutex.c
	nonportable.c
	private.c
	rwlock.c
	sched.c
	semaphore.c
	signal.c
	spin.c
	sync.c
	tsd.c)
	
set(PTHREAD_RESOURCES
	version.rc)
	
set(PTHREAD_HEADERS
	config.h
	implement.h
	need_errno.h)

set(PTHREAD_STATIC_SOURCES
	pthread_attr_init.c
	pthread_attr_destroy.c
	pthread_attr_getdetachstate.c
	pthread_attr_setdetachstate.c
	pthread_attr_getstackaddr.c
	pthread_attr_setstackaddr.c
	pthread_attr_getstacksize.c
	pthread_attr_setstacksize.c
	pthread_attr_getscope.c
	pthread_attr_setscope.c
	pthread_attr_setschedpolicy.c
	pthread_attr_getschedpolicy.c
	pthread_attr_setschedparam.c
	pthread_attr_getschedparam.c
	pthread_attr_setinheritsched.c
	pthread_attr_getinheritsched.c
	pthread_barrier_init.c
	pthread_barrier_destroy.c
	pthread_barrier_wait.c
	pthread_barrierattr_init.c
	pthread_barrierattr_destroy.c
	pthread_barrierattr_setpshared.c
	pthread_barrierattr_getpshared.c
	pthread_setcancelstate.c
	pthread_setcanceltype.c
	pthread_testcancel.c
	pthread_cancel.c
	cleanup.c
	pthread_condattr_destroy.c
	pthread_condattr_getpshared.c
	pthread_condattr_init.c
	pthread_condattr_setpshared.c
	pthread_cond_destroy.c
	pthread_cond_init.c
	pthread_cond_signal.c
	pthread_cond_wait.c
	create.c
	dll.c
	autostatic.c
	errno.c
	pthread_exit.c
	fork.c
	global.c
	pthread_mutex_init.c
	pthread_mutex_destroy.c
	pthread_mutexattr_init.c
	pthread_mutexattr_destroy.c
	pthread_mutexattr_getpshared.c
	pthread_mutexattr_setpshared.c
	pthread_mutexattr_settype.c
	pthread_mutexattr_gettype.c
	pthread_mutexattr_setrobust.c
	pthread_mutexattr_getrobust.c
	pthread_mutex_lock.c
	pthread_mutex_timedlock.c
	pthread_mutex_unlock.c
	pthread_mutex_trylock.c
	pthread_mutex_consistent.c
	pthread_mutexattr_setkind_np.c
	pthread_mutexattr_getkind_np.c
	pthread_getw32threadhandle_np.c
	pthread_getunique_np.c
	pthread_delay_np.c
	pthread_num_processors_np.c
	pthread_win32_attach_detach_np.c
	pthread_equal.c
	pthread_getconcurrency.c
	pthread_once.c
	pthread_self.c
	pthread_setconcurrency.c
	pthread_rwlock_init.c
	pthread_rwlock_destroy.c
	pthread_rwlockattr_init.c
	pthread_rwlockattr_destroy.c
	pthread_rwlockattr_getpshared.c
	pthread_rwlockattr_setpshared.c
	pthread_rwlock_rdlock.c
	pthread_rwlock_wrlock.c
	pthread_rwlock_unlock.c
	pthread_rwlock_tryrdlock.c
	pthread_rwlock_trywrlock.c
	pthread_setschedparam.c
	pthread_getschedparam.c
	pthread_timechange_handler_np.c
	ptw32_is_attr.c
	ptw32_processInitialize.c
	ptw32_processTerminate.c
	ptw32_threadStart.c
	ptw32_threadDestroy.c
	ptw32_tkAssocCreate.c
	ptw32_tkAssocDestroy.c
	ptw32_callUserDestroyRoutines.c
	ptw32_timespec.c
	ptw32_throw.c
	ptw32_getprocessors.c
	ptw32_calloc.c
	ptw32_new.c
	ptw32_reuse.c
	ptw32_rwlock_check_need_init.c
	ptw32_cond_check_need_init.c
	ptw32_mutex_check_need_init.c
	ptw32_semwait.c
	ptw32_relmillisecs.c
	ptw32_MCS_lock.c
	sched_get_priority_max.c
	sched_get_priority_min.c
	sched_setscheduler.c
	sched_getscheduler.c
	sched_yield.c
	sem_init.c
	sem_destroy.c
	sem_trywait.c
	sem_timedwait.c
	sem_wait.c
	sem_post.c
	sem_post_multiple.c
	sem_getvalue.c
	sem_open.c
	sem_close.c
	sem_unlink.c
	signal.c
	pthread_kill.c
	ptw32_spinlock_check_need_init.c
	pthread_spin_init.c
	pthread_spin_destroy.c
	pthread_spin_lock.c
	pthread_spin_unlock.c
	pthread_spin_trylock.c
	pthread_detach.c
	pthread_join.c
	pthread_key_create.c
	pthread_key_delete.c
	pthread_setspecific.c
	pthread_getspecific.c
	w32_CancelableWait.c)

#=======================================================
# TARGET BUILD
#=======================================================		
set(CMAKE_DEBUG_POSTFIX d)
include_directories(${PROJECT_SOURCE_DIR})
add_library(pthreads SHARED ${PTHREAD_SOURCES} ${PTHREAD_API} ${PTHREAD_RESOURCES})	
set_target_properties(pthreads PROPERTIES PUBLIC_HEADER "${PTHREAD_API}")