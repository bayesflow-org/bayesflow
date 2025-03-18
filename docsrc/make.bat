@ECHO OFF

pushd %~dp0

REM Command file for sphinx-polyversion documentation

echo.Warning: This make.bat was not tested. If you encounter errors, please
echo.refer to Makefile and open an issue.

if "%1" == "" goto help
if "%1" == "docs" goto docs
if "%1" == "docs-sequential" goto docssequential
if "%1" == "local" goto local
if "%1" == "clean" goto clean
if "%1" == "clean-all" goto cleanall
if "%1" == "view-docs" goto viewdocs

:help
echo.Please specify a command (local, docs, docs-sequential, clean, clean-all)
goto end

:docssequential
sphinx-polyversion --local poly.py
echo.Copying docs to ../docs
del /q /s ..\docs\*
xcopy /y /s _build_polyversion\ ..\docs
xcopy /y .nojekyll ..\docs\.nojekyll
del /q /s _build_polyversion
goto end

:docssequential
sphinx-polyversion --sequential poly.py
echo.Copying docs to ../docs
del /q /s ..\docs\*
xcopy /y /s _build_polyversion\ ..\docs
xcopy /y .nojekyll ..\docs\.nojekyll
del /q /s _build_polyversion
goto end

:docs
sphinx-polyversion poly.py
echo.Copying docs to ../docs
del /q /s ..\docs\*
xcopy /y /s _build_polyversion\ ..\docs
xcopy /y .nojekyll ..\docs\.nojekyll
del /q /s _build_polyversion
goto end

:viewdocs
echo.Serving the contents of '../docs'... (open the link below to view).
echo.Interrupt with Ctrl+C.
python -m http.server -d ../docs -b 127.0.0.1 8090
goto end

:clean
del /q /s ..\docs\*
rmdir /q /s _build
rmdir /q /s %BUILDDIR%
rmdir /q /s _build_polyversion
rmdir /q /s source\_examples
del /q /s source\contributing.md
del /q /s source\installation.rst
goto end

:cleanall
del /q /s .docs_venvs .bf_doc_gen_venv
goto clean

:end
popd
