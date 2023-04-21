all:
	@echo 'Try `make vi`'

vi:
	vim \
		Makefile \
		README.md \
		settings.sh \
		data/exif-img-tags-settable \
		pg1-video-to-frames \
		pg2-get-blur-values \
		pg3-blur-review \
		pg4-blur-approve \
		pg5-pick-subset \
		pg6-convert-to-pg \
		pg7-set-exif-from-reference \
		pylib/kbnb.py \
		pylib/bansi.py \
		utils/bansi.sh \
		~/bin/histotext \
		utils/blur-detect.py \
		utils/video-to-frames \
		utils/fn.readcountdown \
		utils/face-mask.py
