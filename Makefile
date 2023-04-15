all:
	@echo 'Try `make vi`'

vi:
	vim \
		Makefile \
		README.md \
		settings.sh \
		pg1-video-to-frames \
		pg2-get-blur-values \
		pg3-blur-review \
		pg4-blur-approve \
		pylib/kbnb.py \
		pylib/bansi.py \
		~/bin/histotext \
		utils/blur-detect.py \
		utils/video-to-frames \
		utils/fn.readcountdown
