#!/bin/sh
# Self-contained submission package: LaTeX sources, class files, figures (PDF, numbered as in the article),
# highlights and compiled PDF. Usage: sh package.sh
set -e
cd "$(dirname "$0")"
PKG=RESS_submission
rm -rf "$PKG" RESS_submission.zip
mkdir -p "$PKG/figures"
cp manuscript.tex sec*.tex tab_*.tex refs.tex refs_all.tex cas-dc.cls cas-common.sty highlights.txt build_tex.py manuscript.pdf "$PKG/"
# source figure file -> figure number in the article
for pair in Fig1_method:fig1 Fig2_validity:fig2 Fig3_conditional:fig3 Fig8_robustness:fig4 Fig6_censoring:fig5 Fig7_cvplus_M:fig6 Fig5_transfer:fig7; do
  src=${pair%%:*}; dst=${pair##*:}
  cp "../figures/$src.pdf" "$PKG/figures/$dst.pdf"
  sed -i "s#figures/$src.pdf#figures/$dst.pdf#g" "$PKG"/sec*.tex
done
(cd "$PKG" && pdflatex -interaction=nonstopmode manuscript.tex > /dev/null && pdflatex -interaction=nonstopmode manuscript.tex > /dev/null && echo "package compiles: $(pdfinfo manuscript.pdf | grep Pages)")
grep -h "includegraphics" "$PKG"/sec*.tex | sed 's/.*{figures/  figures/; s/}.*//'
zip -qr RESS_submission.zip "$PKG" -x "*.aux" "*.log" "*.out" "*.abs" "*.spl"
echo "written RESS_submission.zip"
