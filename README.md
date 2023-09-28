# zwartlab-viewer

[![License BSD-3](https://img.shields.io/pypi/l/zwartlab-viewer.svg?color=green)](https://github.com/pnm4sfix/zwartlab-viewer/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/zwartlab-viewer.svg?color=green)](https://pypi.org/project/zwartlab-viewer)
[![Python Version](https://img.shields.io/pypi/pyversions/zwartlab-viewer.svg?color=green)](https://python.org)
[![tests](https://github.com/pnm4sfix/zwartlab-viewer/workflows/tests/badge.svg)](https://github.com/pnm4sfix/zwartlab-viewer/actions)
[![codecov](https://codecov.io/gh/pnm4sfix/zwartlab-viewer/branch/main/graph/badge.svg)](https://codecov.io/gh/pnm4sfix/zwartlab-viewer)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/zwartlab-viewer)](https://napari-hub.org/plugins/zwartlab-viewer)

A simple plugin to visualise and perform registration on 2P datasets created in Maarten Zwart's lab

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation
    conda create -n ZwartlabViewer python=3.10
    
    conda activate ZwartlabViewer

    conda install -c "nvidia/label/cuda-11.7.0" cuda

    conda install -c conda-forge cupy

    

You can install `zwartlab-viewer` via [pip]:

    pip install napari["all"]

    pip install zwartlab-viewer

    pip install git+https://github.com/dipy/cudipy.git

    pip install git+https://github.com/lukasz-migas/napari-plot.git

    

To install latest development version :

    pip install git+https://github.com/pnm4sfix/zwartlab-viewer.git


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"zwartlab-viewer" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/pnm4sfix/zwartlab-viewer/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
