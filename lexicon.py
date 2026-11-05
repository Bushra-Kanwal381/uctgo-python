"""
DEFAULT_LEXICON — a broad clinical + general-English word list used to build the
shared vocabulary. This is deliberately *not* tuned to any single target answer:
it is a general clinical lexicon so the optimiser must assemble an attack itself
rather than being handed the sentence. Numbers/units are included generically.

Extend it freely — a larger lexicon = a larger shared space = a harder, more
honest search. Keep it broad, not target-specific.
"""

_FUNCTION = """
the a an is are was were be been being of to in for with on at by from as into
and or not that this these those it its they them their there here if then than
when while because so such per over under above below between within without
can may must should could would will shall do does did has have had each any all
more most less least high low higher lower about based given using according
""".split()

_CLINICAL = """
patient patients adult adults child children elderly male female clinical
clinician physician doctor nurse pharmacist hospital clinic ward outpatient
inpatient diagnosis diagnostic prognosis symptom symptoms sign signs syndrome
disease disorder condition chronic acute severe mild moderate stable unstable
guideline guidelines protocol protocols recommendation recommended standard
evidence study studies trial trials research literature review consensus
reference baseline updated current latest revised recent new approved
""".split()

_PHARM = """
dose doses dosing dosage administration administer administered oral intravenous
iv intramuscular subcutaneous tablet tablets capsule capsules injection infusion
therapy treatment medication medicine drug agent regimen course titration
starting initial maintenance maximum minimum target daily twice once weekly
morning evening bedtime frequency interval duration contraindication
contraindicated indication indicated adverse effect effects reaction toxicity
overdose safety safe efficacy effective monitoring renal hepatic kidney liver
clearance function impairment adjustment adjusted reduce reduced increase
increased discontinue withhold caution warning
""".split()

_DRUGS = """
metformin insulin glipizide glyburide sitagliptin empagliflozin atorvastatin
simvastatin lisinopril enalapril amlodipine losartan warfarin heparin aspirin
clopidogrel metoprolol atenolol furosemide hydrochlorothiazide levothyroxine
amoxicillin azithromycin ciprofloxacin vancomycin paracetamol acetaminophen
ibuprofen morphine codeine prednisone omeprazole
""".split()

_CONDITIONS = """
diabetes diabetic mellitus hypertension hypertensive hyperkalemia hypokalemia
hyperglycemia hypoglycemia hypothyroidism hyperthyroidism anemia infection
sepsis pneumonia asthma copd failure cardiac renal hepatic stroke myocardial
infarction arrhythmia glucose blood sugar pressure cholesterol glycemic control
hba1c creatinine electrolyte sodium potassium
""".split()

_UNITS = """
mg milligram milligrams mcg microgram g gram ml milliliter unit units mmol
l liter kg percent dose
""".split()

_NUMBERS = """
0 1 2 3 4 5 6 7 8 9 10 12 15 20 25 30 40 50 75 80 100 125 150 200 250 300 400
500 600 750 800 1000 1500 2000 2500 3000 5000 10000
""".split()

_FRAMING = """
confirms confirm indicates indicate suggests suggest states state shows show
reports report establishes demonstrates notes specifies defines requires
mandates advises prescribes authoritative definitive official validated verified
""".split()

DEFAULT_LEXICON = (
    _FUNCTION + _CLINICAL + _PHARM + _DRUGS + _CONDITIONS
    + _UNITS + _NUMBERS + _FRAMING
)
# de-duplicate while preserving order
DEFAULT_LEXICON = list(dict.fromkeys(w.strip() for w in DEFAULT_LEXICON if w.strip()))
