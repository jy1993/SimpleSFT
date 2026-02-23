from llm_utils import read_json, to_json
import os
import json
from collections import Counter
import pandas as pd
from collections import defaultdict
import argparse
from tqdm import tqdm
import random
from pprint import pprint

parser = argparse.ArgumentParser()
parser.add_argument('--result_path', type=str, default=None)
parser.add_argument('--eval_result_save_path', type=str, default=None)
parser.add_argument('--bad_case_save_path', type=str, default=None)
args = parser.parse_args()

config = {
	'coord': 20
}

def locate_in_string(string, sub_string):
	index = []
	for i in range(len(string)):
		if string[i:i+len(sub_string)] == sub_string:
			index.append(i)
	return index

def parse_tool_call(tool_call):
	tc_start = locate_in_string(tool_call, '<tool_call>')
	tc_end = locate_in_string(tool_call, '</tool_call>')
	tc_actions = []
	for s, e in zip(tc_start, tc_end):
		try:
			tc_actions.append(json.loads(tool_call[s+len('<tool_call>'):e]))
		except:
			pass
	return tc_actions

def list_equal(alist, blist):
	if len(alist) != len(blist):
		return False, 'length mismatch'
	if any(a != b for a, b in zip(alist, blist)):
		return False, 'wrong'
	return True, None

def value_match(value1, value2, para_type):
	if para_type == 'coord':
		if abs(value1 - value2) < config['coord']:
			return True
	elif para_type == 'enum':
		if value1 == value2:
			return True
	elif para_type == 'text':
		return True
	return False

def dict_equal(dict1, dict2):
	if 'name' not in dict1 or 'name' not in dict2:
		return False, 'missing tool name'
	name1 = dict1['name']
	name2 = dict2['name']
	if name1 != name2:
		return False, f'tool name mismatch for {name1}'
	p1 = dict1['arguments']
	p2 = dict2['arguments']
	if len(p1) != len(p2):
		return False, 'parameter number mismatch'
	for k, v in p1.items():
		if k not in p2:
			return False, 'parameter name mismatch'
		if k in ['x', 'y']:
			para_type = 'coord'
		elif k == 'text':
			para_type = 'text'
		else:
			para_type = 'enum'
		if not value_match(p2[k], v, para_type):
			return False, f'parameter value mismatch for {k}'
	return True, None

def eval_one(gt, pred):
	gt_actions = parse_tool_call(gt)
	pred_actions = parse_tool_call(pred)
	assert len(gt_actions) == 1
	if len(pred_actions) != 1:
		return 'the number of tool calls mismatch'
	for gt_action, pred_action in zip(gt_actions, pred_actions):
		tool_call_match, wrong_reason = dict_equal(gt_action, pred_action)
		if not tool_call_match:
			return wrong_reason
	return 'match'

data = read_json(args.result_path)
all_eval_result = []
print('number of test data: ', len(data))
for row in tqdm(data):
	flag = eval_one(row['gt']['content'][0]['text'], row['pred'])
	all_eval_result.append({
		'episode_id': row['episode_id'],
		'query': row['input']['content'],
		'gt': row['gt']['content'], 
		'pred': row['pred'], 
		'result': flag})
to_json(all_eval_result, args.eval_result_save_path)
to_json([row for row in all_eval_result if row['result'] != 'match'], args.bad_case_save_path)
ct = Counter([result['result'] for result in all_eval_result])
pprint(ct)
acc = ct['match'] / sum(ct.values())
print('acc: %.3f' % acc)