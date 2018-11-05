# Building the documentation

## Overview

The API documentation is auto generated using
[Sphinx](http://www.sphinx-doc.org). It uses the in-line docstrings in python
code to generate the API documentation. Specifically, we opted to follow the
Google docstrings as defined 
[here](https://github.com/google/styleguide/blob/gh-pages/pyguide.md), so
anyone intending to add/modify new in-line strings should strictly attach to
those guidelines.

Additonally to the auto-generated html pages, we added some static pages that
show how to use the tool and other useful docs. These are:

- TODO: Add the custom list

Please be careful to do not delete these.

## Sphinx in a nutshell

Sphinx requires a set of *sources* that are `reStructuredText` files. For us,
these files are stored in the `docs/source` folder and some of them are static
and some others are auto generated using the `sphix-apidoc` command-line tool.

After that, the build's sphinx tool takes these files and generates an html
version of them in a *build directory* (for us: `docs/build/`) ready to deploy
as an static web site.

## Steps

In order to build the documentation, run the next shell commands as listed. The
so called `GIT_STORAGE` refers to the place where the git project has been
cloned.

```bash
# 1. Go to the auto-ml directory
cd ${GIT_STORAGE}/auto-ml

# 2. Run the next command
sphinx-apidoc -o `pwd`/docs/source -f automl

# 3. Go to the docs directory
cd ${GIT_STORAGE}/auto-ml/docs

# 3. Clean the docs
make clean

# 5. Build the docs
make html
```

The documentation will be available in the directory
`${GIT_STORAGE}/auto-ml/docs/build/html` and the main's file name is
`index.html`.