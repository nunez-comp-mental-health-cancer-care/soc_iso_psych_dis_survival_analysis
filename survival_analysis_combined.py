from preprocesser import preprocess
from lifelines import CoxPHFitter
import os
from globals import RESULTS_DIR

# Mapping from preprocessed column names to interpretable labels
FEAT_RENAME = {
    'age_dx':               'age_at_diagnosis',
    'male':                 'female',           # will be inverted below
    'M1_IV_ALL_CAT':        'metastatic',
    'LiveAlone':            'lives_alone',
    'NoAssistance':         'no_iadl_assistance',
    'NoRegularContact':     'no_regular_contact',
    'LostSpousePartner':    'lost_partner',
    'NoEmotionalSupport':   'no_emotional_support',
    'DepressionScore_CAT2': 'depression',
    'AnxietyScore_CAT2':    'anxiety',
}

CPH_COLS = ['observed_mo', 'died'] + list(FEAT_RENAME.keys())


def run_combined_cph(results_dir=RESULTS_DIR):
    output_dir = os.path.join(results_dir, 'combined_analysis')
    os.makedirs(output_dir, exist_ok=True)

    df = preprocess()
    df = df.copy()[CPH_COLS]
    print(f"CPH analysis includes {len(df)} patients")

    # Invert 'male' -> 'female' so True = female
    df['male'] = ~df['male']
    df = df.rename(columns=FEAT_RENAME)

    cph = CoxPHFitter()
    cph.fit(df, 'observed_mo', event_col='died')
    cph.print_summary()

    out_path = os.path.join(output_dir, 'combined_cph.csv')
    cph.summary.to_csv(out_path)
    print(f"CPH summary saved to {out_path}")


if __name__ == '__main__':
    run_combined_cph()
    print("Analysis complete!")