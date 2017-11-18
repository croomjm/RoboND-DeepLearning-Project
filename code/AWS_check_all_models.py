import os
import json
import time

from utils import scoring_utils
from utils import model_tools
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D

class model_results(object):
    def __init__(self,weight_file_name):
        self.data = {'File Name': weight_file_name, 'Scores':{}}
        directory = os.listdir('../data/weights/')
        file_name = 'results_{}.json'.format(weight_file_name)
        if file_name in directory:
            print('Opening existing model results file.')
            with open('../data/weights/' + file_name, 'r') as fp:
                self.data = json.load(fp)
        else:
            #set model parameters based on file name
            self.parse_file_name(weight_file_name)

            self.model = self.load_model(weight_file_name)
            predictions = self.write_predictions(self.model, weight_file_name)
            self.score_model(self.model, predictions)

            #save results data to json file in weights folder
            self.save_model_results()

            #export processed sample interpreted images to folder
            self.export_run(weight_file_name ,self.model)

    def save_model_results(self):
        print('Saving model results for {} to file.'.format(self.data['File Name']))
        with open('../data/weights/results_{}.json'.format(self.data['File Name']), 'w') as fp:
            json.dump(self.data, fp, indent = 4, sort_keys = True)

    def parse_file_name(self, weight_file_name):
        keys = ['Learning Rate', 'Batch Size', 'Epochs', 'Steps per Epoch', 'Validation Steps', 'Optimizer', 'Date Code']
        vals = weight_file_name.split('_')

        self.data['Learning Rate'] = float('0.' + vals[1])
        self.data['Batch Size'] = int(vals[3])
        self.data['Epochs'] = int(vals[5])
        self.data['Steps per Epoch'] = int(vals[7])
        self.data['Validation Steps'] = int(vals[10])
        self.data['Optimizer'] = vals[13]
        self.data['Date Created'] = vals[15]

    def load_model(self, weight_file_name):
        #loads weight file located at ../data/weights/
        #returns corresponding model
        print('Loading model...')
        model = model_tools.load_network(weight_file_name)
        return model

    def export_run(self, run_num, model):
        val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                                run_num,'patrol_with_targ', 'sample_evaluation_data') 

        val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                                run_num,'patrol_non_targ', 'sample_evaluation_data') 

        val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                                run_num,'following_images', 'sample_evaluation_data')

    def write_predictions(self, model, weight_file_name):
        #write predictions for various scenarios to file, return path to files for each scenario

        run_num = 'run_{}'.format(weight_file_name)

        val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                                run_num,'patrol_with_targ', 'sample_evaluation_data') 

        val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model, 
                                                run_num,'patrol_non_targ', 'sample_evaluation_data') 

        val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                                run_num,'following_images', 'sample_evaluation_data')

        return [val_with_targ, pred_with_targ, val_no_targ, pred_no_targ, val_following, pred_following]

    def score_model(self, model, predictions):
        val_with_targ, pred_with_targ, val_no_targ, pred_no_targ, val_following, pred_following = predictions

        print('Scoring Model...')
        # Scores for while the quad is following behind the target. 
        true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)
        self.data['Scores']['Following Target'] = {}
        self.data['Scores']['Following Target']['True Positives'] = true_pos1
        self.data['Scores']['Following Target']['False Positives'] = false_pos1
        self.data['Scores']['Following Target']['False Negatives'] = false_neg1
        self.data['Scores']['Following Target']['IOU'] = iou1

        # Scores for images while the quad is on patrol and the target is not visable
        true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)
        self.data['Scores']['No Target'] = {}
        self.data['Scores']['No Target']['True Positives'] = true_pos2
        self.data['Scores']['No Target']['False Positives'] = false_pos2
        self.data['Scores']['No Target']['False Negatives'] = false_neg2
        self.data['Scores']['No Target']['IOU'] = iou2

        # This score measures how well the neural network can detect the target from far away
        true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)
        self.data['Scores']['Far from Target'] = {}
        self.data['Scores']['Far from Target']['True Positives'] = true_pos3
        self.data['Scores']['Far from Target']['False Positives'] = false_pos3
        self.data['Scores']['Far from Target']['False Negatives'] = false_neg3
        self.data['Scores']['Far from Target']['IOU'] = iou3

        # Sum all the true positives, etc from the three datasets to get a weight for the score
        self.data['Scores']['Overall'] = {}
        self.data['Scores']['Overall']['True Positives'] = true_pos1 + true_pos2 + true_pos3
        self.data['Scores']['Overall']['False Positives'] = false_pos1 + false_pos2 + false_pos3
        self.data['Scores']['Overall']['False Negatives'] = false_neg1 + false_neg2 + false_neg3

        true_pos = true_pos1 + true_pos2 + true_pos3
        false_neg = false_neg1 + false_neg2 + false_neg3
        false_pos = false_pos1 + false_pos2 + false_pos3

        weight = true_pos/(true_pos+false_neg+false_pos)
        self.data['Scores']['Overall']['Weight'] = weight

        # The IoU for the dataset that never includes the hero is excluded from grading
        final_IoU = (iou1 + iou3)/2
        self.data['Scores']['Overall']['IOU'] = final_IoU

        # And the final grade score is 
        final_score = final_IoU * weight
        self.data['Scores']['Overall']['Score'] = final_score

        print('Model Results for {}:'.format(self.data['File Name']))
        print('    Weight: {}'.format(weight))
        print('    Final IoU: {}'.format(final_IoU))
        print('    Final Score: {}'.format(final_score))

def sort_models_by_score(results_list):
    return sorted(results_list, key = lambda result: result['Scores']['Overall']['Score'])

def generate_github_md_results_tables(results_list):
    print('Saving model results table to file.')

    content = generate_github_md_summary_results_table(results_list)
    content.append('\n')
    content.extend(generate_github_md_detailed_results_table(results_list))

    with open('../project_submission/results_table_{}'.format(time.strftime("%Y%m%d-%H%M%S")), 'w') as text_file:
        text_file.write('\n'.join(content))

def generate_github_md_summary_results_table(results_list):
    #table listing parameters of models and overall scores
    header_cols = ['', 'Learning Rate', 'Batch Size', 'Epochs', 'Steps per Epoch',
                   'Validation Steps', 'Optimizer', 'IOU', 'Score']
    header_row = '| ' + ' | '.join(header_cols) + ' |'
    header_underline = '|:---:'*len(header_cols) + '|'

    content = [header_cols, header_underline]
    for i,r in enumerate(results_list):
        row = ''
        for c in header_cols:
            if c == '':
                row += '| ' + str(i) + ' |'
            elif c == 'IOU' or c == 'Score':
                row += '| ' + '{0:0.3f}'.format(r['Scores']['Overall'][c]) + ' |'
            else:
                if isinstance(r[c], float):
                    item = round(r[c], 3)
                else:
                    item = str(r[c])
                row += '| ' + item + ' |'
        content.append(row)

    return content

def generate_github_md_detailed_results_table(results_list):
    #table with false negatives, false positives, etc. but not model parameters
    header_cols = ['',
                    'Overall\nScore'
                    'Following Target\nTrue Positives','Following Target\nFalse Positives','Following Target\nFalse Negatives',
                    'No Target\nTrue Positives','No Target\nFalse Positives','No Target\nFalse Negatives',
                    'Far from Target\nTrue Positives','Far from Target\nFalse Positives','Far from Target\nFalse Negatives',
                    ]

    header_row = '| ' + ' | '.join(header_cols) + ' |'
    header_underline = '|:---:'*len(header_cols) + '|'
    content = [header_cols, header_underline]

    for i,r in enumerate(results_list):
        row = ''
        for c in header_cols:
            if c == '':
                row += '| ' + str(i) + ' |'
            else:
                key1, key2 = c.split('\n')
                if isinstance(r['Scores'][key1][key2], float):
                    item = round(r['Scores'][key1][key2], 3)
                else:
                    item = str(r['Scores'][key1][key2])
                row += '| ' + item + ' |'
        content.append(row)

    return content

def main():
    print('Finding weights files.')
    directory = os.listdir('../data/weights')
    weight_files = []
    results_list = []
    for f in directory:
        parsed_file_name = f.split('_')
        if parsed_file_name[0] == 'weights' and len(parsed_file_name) == 16:
            weight_files.append(f)

    print('Weight Files:\n{}'.format('\n'.join(weight_files)))
    for f in weight_files:
        print('Calculating results for {}'.format(f))
        results = model_results(f)
        results_list.append(results.data)

    print('Sorting results by score.')
    results_list = sort_models_by_score(results_list)

    print('Generating github md results summary tables.')
    generate_github_md_results_tables(results_list)

if __name__ == '__main__':
    print('Starting main function.')
    main()











