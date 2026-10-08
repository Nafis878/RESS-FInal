"""Assemble the manuscript: sections/*.md -> manuscript.md (numbered citations, reference list) -> manuscript.docx
(pandoc) -> manuscript.pdf (LibreOffice). Usage: python build.py"""
import os, re, subprocess, glob

HERE = os.path.dirname(os.path.abspath(__file__))
REFS = {
    'barlow1960': 'Barlow R, Hunter L. Optimum preventive maintenance policies. Operations Research 1960;8(1):90–100. https://doi.org/10.1287/opre.8.1.90',
    'dejonge2020': 'de Jonge B, Scarf PA. A review on maintenance optimization. European Journal of Operational Research 2020;285(3):805–824. https://doi.org/10.1016/j.ejor.2019.09.047',
    'saxena2008': 'Saxena A, Goebel K, Simon D, Eklund N. Damage propagation modeling for aircraft engine run-to-failure simulation. In: 2008 International Conference on Prognostics and Health Management. IEEE; 2008, p. 1–9. https://doi.org/10.1109/PHM.2008.4711414',
    'saxena2010': 'Saxena A, Celaya J, Saha B, Saha S, Goebel K. Metrics for offline evaluation of prognostic performance. International Journal of Prognostics and Health Management 2010;1(1). https://doi.org/10.36001/ijphm.2010.v1i1.1336',
    'ramasso2014': 'Ramasso E, Saxena A. Performance benchmarking and analysis of prognostic methods for CMAPSS datasets. International Journal of Prognostics and Health Management 2014;5(2). https://doi.org/10.36001/ijphm.2014.v5i2.2236',
    'depater2022': 'de Pater I, Reijns A, Mitici M. Alarm-based predictive maintenance scheduling for aircraft engines with imperfect remaining useful life prognostics. Reliability Engineering & System Safety 2022;221:108341. https://doi.org/10.1016/j.ress.2022.108341',
    'kamariotis2024': 'Kamariotis A, Tatsis K, Chatzi E, Goebel K, Straub D. A metric for assessing and optimizing data-driven prognostic algorithms for predictive maintenance. Reliability Engineering & System Safety 2024;242:109723. https://doi.org/10.1016/j.ress.2023.109723',
    'nguyen2019': 'Nguyen KTP, Medjaher K. A new dynamic predictive maintenance framework using deep learning for failure prognostics. Reliability Engineering & System Safety 2019;188:251–262. https://doi.org/10.1016/j.ress.2019.03.018',
    'mitici2023': 'Mitici M, de Pater I, Barros A, Zeng Z. Dynamic predictive maintenance for multiple components using data-driven probabilistic RUL prognostics: The case of turbofan engines. Reliability Engineering & System Safety 2023;234:109199. https://doi.org/10.1016/j.ress.2023.109199',
    'ariaschao2021': 'Arias Chao M, Kulkarni C, Goebel K, Fink O. Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics. Data 2021;6(1):5. https://doi.org/10.3390/data6010005',
    'severson2019': 'Severson KA, Attia PM, Jin N, Perkins N, Jiang B, Yang Z, Chen MH, Aykol M, Herring PK, Fraggedakis D, Bazant MZ, Harris SJ, Chueh WC, Braatz RD. Data-driven prediction of battery cycle life before capacity degradation. Nature Energy 2019;4(5):383–391. https://doi.org/10.1038/s41560-019-0356-8',
    'nectoux2012': 'Nectoux P, Gouriveau R, Medjaher K, Ramasso E, Chebel-Morello B, Zerhouni N, Varnier C. PRONOSTIA: An experimental platform for bearings accelerated degradation tests. In: IEEE International Conference on Prognostics and Health Management (PHM\'12), Denver, CO; 2012. https://hal.science/hal-00719503',
    'pedregosa2011': 'Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 2011;12:2825–2830.',
    'clopper1934': 'Clopper CJ, Pearson ES. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika 1934;26(4):404–413. https://doi.org/10.1093/biomet/26.4.404',
    'holm1979': 'Holm S. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 1979;6(2):65–70.',
    'vovk2005': 'Vovk V, Gammerman A, Shafer G. Algorithmic Learning in a Random World. New York: Springer; 2005. https://doi.org/10.1007/b106715',
    'lei2018': "Lei J, G'Sell M, Rinaldo A, Tibshirani RJ, Wasserman L. Distribution-free predictive inference for regression. Journal of the American Statistical Association 2018;113(523):1094–1111. https://doi.org/10.1080/01621459.2017.1307116",
    'angelopoulos2023': 'Angelopoulos AN, Bates S. Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning 2023;16(4):494–591. https://doi.org/10.1561/2200000101',
    'javanmardi2023': 'Javanmardi A, Hüllermeier E. Conformal prediction intervals for remaining useful lifetime estimation. International Journal of Prognostics and Health Management 2023;14(2). https://doi.org/10.36001/ijphm.2023.v14i2.3417',
    'robinson2026': 'Robinson CD. Remaining useful life estimation for aircraft engines with risk-aware prediction intervals via conformalized quantile regression. International Journal of Prognostics and Health Management 2026 (published online April 2026).',
    'wang2025robustuq': 'Wang W, Wang Z, Cai Z, Hu C, Si S. Robust uncertainty quantification for online remaining useful life prediction with randomly missing and partially faulty sensor data. Reliability Engineering & System Safety 2025;262.',
    'romano2019': 'Romano Y, Patterson E, Candès E. Conformalized quantile regression. In: Advances in Neural Information Processing Systems 32 (NeurIPS 2019); 2019.',
    'candes2023': 'Candès E, Lei L, Ren Z. Conformalized survival analysis. Journal of the Royal Statistical Society Series B 2023;85(1):24–45. https://doi.org/10.1093/jrsssb/qkac004',
    'angelopoulos2024': 'Angelopoulos AN, Bates S, Fisch A, Lei L, Schuster T. Conformal risk control. In: International Conference on Learning Representations (ICLR); 2024.',
    'gupta2022': 'Gupta C, Kuchibhotla AK, Ramdas A. Nested conformal prediction and quantile out-of-bag ensemble methods. Pattern Recognition 2022;127:108496. https://doi.org/10.1016/j.patcog.2021.108496',
    'tibshirani2019': 'Tibshirani RJ, Foygel Barber R, Candès E, Ramdas A. Conformal prediction under covariate shift. In: Advances in Neural Information Processing Systems 32 (NeurIPS 2019); 2019.',
    'barber2023': 'Barber RF, Candès EJ, Ramdas A, Tibshirani RJ. Conformal prediction beyond exchangeability. Annals of Statistics 2023;51(2):816–845. https://doi.org/10.1214/23-AOS2276',
    'vovk2012': 'Vovk V. Conditional validity of inductive conformal predictors. In: Proceedings of the Asian Conference on Machine Learning, PMLR 25; 2012, p. 475–490.',
    'tango1998': 'Tango T. Equivalence test and confidence interval for the difference in proportions for the paired-sample design. Statistics in Medicine 1998;17(8):891–908.',
    'schuirmann1987': 'Schuirmann DJ. A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. Journal of Pharmacokinetics and Biopharmaceutics 1987;15(6):657–680. https://doi.org/10.1007/BF01068419',
    'prior': 'Anonymous (authors of the present study). What does an end-of-life alarm score measure? Separating prediction accuracy, alarm timing, maintenance cost and transfer. Earlier manuscript version, 2026 (superseded by this article; numbers attributed to it are prior results).',
}


def main():
    parts = sorted(glob.glob(os.path.join(HERE, 'sections', '*.md')))
    text = '\n\n'.join(open(p, encoding='utf-8').read() for p in parts)
    order = []
    def cite(m):
        keys = [k.strip().lstrip('@') for k in m.group(1).split(';')]
        nums = []
        for k in keys:
            if k not in REFS: raise KeyError(f'missing reference {k}')
            if k not in order: order.append(k)
            nums.append(order.index(k) + 1)
        nums = sorted(nums); out = []; i = 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
            out.append(f'{nums[i]}–{nums[j]}' if j - i >= 2 else ', '.join(str(n) for n in nums[i:j + 1])); i = j + 1
        return '[' + ', '.join(out) + ']'
    text = re.sub(r'\[(@[^\]]+)\]', cite, text)
    text += '\n\n# References {-}\n\n' + '\n'.join(f'{i + 1}. {REFS[k]}' for i, k in enumerate(order)) + '\n'
    md = os.path.join(HERE, 'manuscript.md'); open(md, 'w', encoding='utf-8').write(text)
    unused = sorted(set(REFS) - set(order))
    if unused: print('unused references:', unused)
    ref = os.path.join(HERE, 'reference.docx')
    cmd = ['pandoc', md, '-o', os.path.join(HERE, 'manuscript.docx'), '--resource-path', HERE + ':' + os.path.dirname(HERE)]
    if os.path.exists(ref): cmd += ['--reference-doc', ref]
    subprocess.run(cmd, check=True)
    # PDF: HTML with MathML printed by Chromium (LibreOffice mangles some OMML constructs)
    html = os.path.join(HERE, 'manuscript.html')
    subprocess.run(['pandoc', md, '-s', '--mathml', '--embed-resources', '--css', os.path.join(HERE, 'print.css'), '-o', html,
                    '--resource-path', HERE + ':' + os.path.dirname(HERE), '--metadata', 'lang=en'], check=True)
    chrome = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
    subprocess.run([chrome, '--headless', '--no-sandbox', '--disable-gpu', '--no-pdf-header-footer', f'--print-to-pdf={os.path.join(HERE, "manuscript.pdf")}',
                    'file://' + html], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print('built manuscript.md, manuscript.docx, manuscript.pdf;', len(order), 'references')


if __name__ == '__main__':
    main()
