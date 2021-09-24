import os

from ExplainaBoard.interpret_eval.tasks.ner.tensoreval import evaluate

test_iob_dir = 'data/test_outputs_iob'
annotation_files = os.listdir(test_iob_dir)

for file in annotation_files:
    print(file)
    input_file = f'{test_iob_dir}/{file}'
    output_file = f'data/evaluation_results/{file.replace(".tsv", ".json")}'
    task = 'ner'
    analysis_type = 'single'
    ci = case = ece = False

    evaluate(task_type=task,
             analysis_type=analysis_type,
             systems=[input_file],
             output=output_file,
             is_print_ci=ci,
             is_print_case=case,
             is_print_ece=ece,
             model_name=file.replace('.tsv',''))
