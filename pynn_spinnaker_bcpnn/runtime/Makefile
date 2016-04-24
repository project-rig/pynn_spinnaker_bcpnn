APP_BUILD_DIRS = $(sort $(dir $(wildcard build/*/)))

all: $(APP_BUILD_DIRS)
	echo $(APP_BUILD_DIRS)
	for d in $(APP_BUILD_DIRS); do (cd $$d; "$(MAKE)") || exit $$?; done

clean: $(APP_BUILD_DIRS)
	for d in $(APP_BUILD_DIRS); do (cd $$d; "$(MAKE)" clean) || exit $$?; done
