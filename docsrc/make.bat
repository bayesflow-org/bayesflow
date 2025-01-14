@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=source
set BUILDDIR=_build

if "%1" == "" goto help
if "%1" == "github" goto github

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

echo.Warning: This make.bat was not tested. If you encounter errors, please
echo.refer to Makefile.

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:dev
python pre-build.py source
%SPHINXBUILD% -M html %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
xcopy /y /s "%BUILDDIR%\html" ..\docs
xcopy /y .nojekyll ..\docs\.nojekyll
goto end

:github
set BF_DOCS_SEQUENTIAL_BUILDS=1
sphinx-polyversion poly.py
xcopy /y /s _build_polyversion\ ..\docs
xcopy /y .nojekyll ..\docs\.nojekyll
goto end

:parallel
set BF_DOCS_SEQUENTIAL_BUILDS=0
sphinx-polyversion poly.py
xcopy /y /s _build_polyversion\ ..\docs
xcopy /y .nojekyll ..\docs\.nojekyll

:clean
del /q /s ..\docs\*
rmdir /q /s _build
rmdir /q /s %BUILDDIR%
rmdir /q /s _build_polyversion
rmdir /q /s source\_examples
del /q /s source\contributing.md
del /q /s source\installation.rst
goto end


:end
popd
