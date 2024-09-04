import pandas as pd

df = pd.read_csv('new_output_qa_pairs.csv')

df = df.drop_duplicates(subset=['question', 'answer', 'chunk', 'pdf_name'])

mapping = {
    'FAQ - Permessi retribuiti': '564 AOS',
    'faq_-_pagina_web_settore_privacy': '762 ARAL',
    'rad_2023_0': '294 ARSS',
    'regolamento2024_0': '296-ARSS',
    'Video istruzioni per congedi e permessi': '78 AOS'
}

df['process_id'] = df['pdf_name'].map(mapping)

df.to_csv('full_data.csv', index=False)
