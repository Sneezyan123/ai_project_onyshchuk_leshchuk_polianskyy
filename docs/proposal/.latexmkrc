#!/usr/bin/env perl
# Конфігурація latexmk: примусово XeLaTeX + BibTeX.
# Overleaf це також підхоплює через "% !TeX program = xelatex" у main.tex.
#
# Свідомо НЕ biber: biber — це окремий важкий процес плюс два зайві проходи
# XeLaTeX через .bcf/.run.xml. На безкоштовному плані Overleaf саме це
# з'їдало більшу частину ліміту часу компіляції.

$pdf_mode = 1;
$pdflatex = 'xelatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';
$bibtex = 'bibtex %O %B';
$bibtex_use = 2;

# Голий `latexmk` без аргументів збирає .tex-файл із поточного каталогу:
# main.tex у корені, example.tex у example/. Рядок навмисно однаковий в обох
# копіях цього файлу — CI перевіряє їх на побайтову ідентичність.
@default_files = glob('*.tex');

$clean_ext = 'bbl blg synctex.gz fdb_latexmk fls';
