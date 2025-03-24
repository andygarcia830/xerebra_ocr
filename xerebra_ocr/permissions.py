import frappe

def check_app_permission():
	"""Check if user has permission to access the app (for showing the app on app screen)"""
	if frappe.session.user == "Administrator":
		return True

	if frappe.db.exists("Xerebra OCR User", {"user": frappe.session.user}):
		return True

	return False
