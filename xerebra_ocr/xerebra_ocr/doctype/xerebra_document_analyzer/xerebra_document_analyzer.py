# Copyright (c) 2025, XAIL and contributors
# For license information, please see license.txt

import frappe
from frappe.utils import get_site_name

from xerebra_ocr.xerebra_ocr.util.ocr_tools import process_file, authenticate

from frappe.model.document import Document


class XerebraDocumentAnalyzer(Document):
	pass

@frappe.whitelist()
def process_upload(filename):
	if type(filename) == type(None) or len(filename) < 1:
		return ''
	authenticate()
	filename = f'{get_site_name(frappe.local.request.host)}/{filename}'
	quota = get_quota()
	limit = quota[0]
	usage = quota[1]
	if limit <= usage:
		frappe.msgprint(
			msg='Usage limit reached for this account. Please contact your sales associate for assistance',
			title='Error',
			# raise_exception=FileNotFoundError
		
		)
		return '<div></div>'
	result = process_file(filename)
	usage += 1
	frappe.db.sql('UPDATE `tabXerebra OCR User` set used= %s WHERE user = %s', (usage, frappe.session.user,))
	return result;

@frappe.whitelist()
def get_quota_display():
	print(f'USER {frappe.session.user}')
	if frappe.session.user == 'Administrator':
		return 'Unlimited'
	result = frappe.db.sql('SELECT quota, used FROM `tabXerebra OCR User` WHERE user = %s', (frappe.session.user,))
	# Retrieve the data
	# ocr_user = frappe.get_doc('Xerebra OCR User', frappe.session.user)
	if (len(result) < 1):
		return 'No Permission'
	print(f'OCR USER {result[0]}')
	quota = result[0]
	return (f'{quota[1]}/{quota[0]} Conversions')

@frappe.whitelist()
def get_quota():
	print(f'USER {frappe.session.user}')
	if frappe.session.user == 'Administrator':
		return (1,0)
	result = frappe.db.sql('SELECT quota, used FROM `tabXerebra OCR User` WHERE user = %s', (frappe.session.user,))
	# Retrieve the data
	# ocr_user = frappe.get_doc('Xerebra OCR User', frappe.session.user)
	print(f'OCR USER {result[0]}')
	return (result[0])