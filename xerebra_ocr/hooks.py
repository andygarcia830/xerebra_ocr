app_name = "xerebra_ocr"
app_title = "Xerebra OCR"
app_publisher = "XAIL"
app_description = "Xerebra OCR"
app_email = "andy@xurpas.com"
app_license = "mit"
# required_apps = []

# Includes in <head>
# ------------------

# include js, css files in header of desk.html
# app_include_css = "/assets/xerebra_ocr/css/xerebra_ocr.css"
# app_include_js = "/assets/xerebra_ocr/js/xerebra_ocr.js"

# include js, css files in header of web template
# web_include_css = "/assets/xerebra_ocr/css/xerebra_ocr.css"
# web_include_js = "/assets/xerebra_ocr/js/xerebra_ocr.js"

# include custom scss in every website theme (without file extension ".scss")
# website_theme_scss = "xerebra_ocr/public/scss/website"

# include js, css files in header of web form
# webform_include_js = {"doctype": "public/js/doctype.js"}
# webform_include_css = {"doctype": "public/css/doctype.css"}

# include js in page
# page_js = {"page" : "public/js/file.js"}

# include js in doctype views
# doctype_js = {"doctype" : "public/js/doctype.js"}
# doctype_list_js = {"doctype" : "public/js/doctype_list.js"}
# doctype_tree_js = {"doctype" : "public/js/doctype_tree.js"}
# doctype_calendar_js = {"doctype" : "public/js/doctype_calendar.js"}

# Svg Icons
# ------------------
# include app icons in desk
# app_include_icons = "xerebra_ocr/public/icons.svg"

# Home Pages
# ----------

# application home page (will override Website Settings)
# home_page = "login"

# website user home page (by Role)
# role_home_page = {
# 	"Role": "home_page"
# }

# Generators
# ----------

# automatically create page for each record of this doctype
# website_generators = ["Web Page"]

# Jinja
# ----------

# add methods and filters to jinja environment
# jinja = {
# 	"methods": "xerebra_ocr.utils.jinja_methods",
# 	"filters": "xerebra_ocr.utils.jinja_filters"
# }

# Installation
# ------------

# before_install = "xerebra_ocr.install.before_install"
# after_install = "xerebra_ocr.install.after_install"

# Uninstallation
# ------------

# before_uninstall = "xerebra_ocr.uninstall.before_uninstall"
# after_uninstall = "xerebra_ocr.uninstall.after_uninstall"

# Integration Setup
# ------------------
# To set up dependencies/integrations with other apps
# Name of the app being installed is passed as an argument

# before_app_install = "xerebra_ocr.utils.before_app_install"
# after_app_install = "xerebra_ocr.utils.after_app_install"

# Integration Cleanup
# -------------------
# To clean up dependencies/integrations with other apps
# Name of the app being uninstalled is passed as an argument

# before_app_uninstall = "xerebra_ocr.utils.before_app_uninstall"
# after_app_uninstall = "xerebra_ocr.utils.after_app_uninstall"

# Desk Notifications
# ------------------
# See frappe.core.notifications.get_notification_config

# notification_config = "xerebra_ocr.notifications.get_notification_config"

# Permissions
# -----------
# Permissions evaluated in scripted ways

# permission_query_conditions = {
# 	"Event": "frappe.desk.doctype.event.event.get_permission_query_conditions",
# }
#
# has_permission = {
# 	"Event": "frappe.desk.doctype.event.event.has_permission",
# }

# DocType Class
# ---------------
# Override standard doctype classes

# override_doctype_class = {
# 	"ToDo": "custom_app.overrides.CustomToDo"
# }

# Document Events
# ---------------
# Hook on document methods and events

# doc_events = {
# 	"*": {
# 		"on_update": "method",
# 		"on_cancel": "method",
# 		"on_trash": "method"
# 	}
# }

# Scheduled Tasks
# ---------------

# scheduler_events = {
# 	"all": [
# 		"xerebra_ocr.tasks.all"
# 	],
# 	"daily": [
# 		"xerebra_ocr.tasks.daily"
# 	],
# 	"hourly": [
# 		"xerebra_ocr.tasks.hourly"
# 	],
# 	"weekly": [
# 		"xerebra_ocr.tasks.weekly"
# 	],
# 	"monthly": [
# 		"xerebra_ocr.tasks.monthly"
# 	],
# }

# Testing
# -------

# before_tests = "xerebra_ocr.install.before_tests"

# Overriding Methods
# ------------------------------
#
# override_whitelisted_methods = {
# 	"frappe.desk.doctype.event.event.get_events": "xerebra_ocr.event.get_events"
# }
#
# each overriding function accepts a `data` argument;
# generated from the base implementation of the doctype dashboard,
# along with any modifications made in other Frappe apps
# override_doctype_dashboards = {
# 	"Task": "xerebra_ocr.task.get_dashboard_data"
# }

# exempt linked doctypes from being automatically cancelled
#
# auto_cancel_exempted_doctypes = ["Auto Repeat"]

# Ignore links to specified DocTypes when deleting documents
# -----------------------------------------------------------

# ignore_links_on_delete = ["Communication", "ToDo"]

# Request Events
# ----------------
# before_request = ["xerebra_ocr.utils.before_request"]
# after_request = ["xerebra_ocr.utils.after_request"]

# Job Events
# ----------
# before_job = ["xerebra_ocr.utils.before_job"]
# after_job = ["xerebra_ocr.utils.after_job"]

# User Data Protection
# --------------------

# user_data_fields = [
# 	{
# 		"doctype": "{doctype_1}",
# 		"filter_by": "{filter_by}",
# 		"redact_fields": ["{field_1}", "{field_2}"],
# 		"partial": 1,
# 	},
# 	{
# 		"doctype": "{doctype_2}",
# 		"filter_by": "{filter_by}",
# 		"partial": 1,
# 	},
# 	{
# 		"doctype": "{doctype_3}",
# 		"strict": False,
# 	},
# 	{
# 		"doctype": "{doctype_4}"
# 	}
# ]

# Authentication and authorization
# --------------------------------

# auth_hooks = [
# 	"xerebra_ocr.auth.validate"
# ]

# Automatically update python controller files with type annotations for this app.
# export_python_type_annotations = True

# default_log_clearing_doctypes = {
# 	"Logging DocType Name": 30  # days to retain logs
# }

add_to_apps_screen = [
	{
		"name": "/app/xerebra-document-analyze",
		"logo": "/assets/xerebra_ocr/X_logo_square.svg",
		"title": "Xerebra OCR",
		"route": "/app/xerebra-document-analyzer",
		"has_permission": "xerebra_ocr.permissions.check_app_permission"
	}
]

fixtures = [
    # export only those records that match the filters from the Role table
    {"dt": "Role", "filters": [["role_name", "like", "Xerebra OCR User"]]},
    {"dt": "Custom DocPerm","filters": [["role","like","Xerebra OCR%"]]},
	{"dt": "Module Profile", "filters": [["module_profile_name", "like", "Xerebra OCR %"]]},
    ]
