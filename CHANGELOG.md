# Change Log
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [v0.0.1] - 2017-06-XX

### Added

*  Python 3 support
*  integration with nosetests

### Changed

*  Broke old API in favour of maintaining `scikit-learn` 0.18 compatibility and `SelectKBest`

### Notes

*  `group_fs` does not work in Python 3 and will require further investigation. 
*  Integration the appropriate `feature_selection` within `scikit-learn` has probably broken sparse filter algorithms