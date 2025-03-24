// Copyright (c) 2025, XAIL and contributors
// For license information, please see license.txt

frappe.ui.form.on("Xerebra Document Analyzer", {
	refresh(frm) {
        frm.disable_save();
        frappe.call({method:'xerebra_ocr.xerebra_ocr.doctype.xerebra_document_analyzer.xerebra_document_analyzer.get_quota_display',
            args:{
            },
            callback: function(response) {
                if (response.message) {
                    console.log(response.message)
                    frm.doc.usage = response.message
                    frm.refresh_field('usage')
                }
                if (frm.doc.usage == 'No Permission') {
                    frm.set_df_property('upload_imageimage_pdf', 'hidden', true)
                }
            }
        })

	},
    before_save: function(frm) {
        console.log('ATTACH '+frm.doc.upload_imageimage_pdf)
        if (!frm.doc.upload_imageimage_pdf || frm.doc.upload_imageimage_pdf == '' ) {
            console.log('INSIDE ATTACH '+frm.doc.upload_imageimage_pdf)
            frm.doc.output = "<div> </div>"
            frm.set_df_property('output','options',frm.doc.output);
            frm.refresh_field('output')
            frappe.validated = false;
            return;
        }
        frm.doc.output = "<div class=\"d-flex justify-content-center\">"+
                    "<div class=\"spinner-border text-primary\" role=\"status\">"+
                    "</div>"
        frm.set_df_property('output','options',frm.doc.output);
        frappe.call({method:'xerebra_ocr.xerebra_ocr.doctype.xerebra_document_analyzer.xerebra_document_analyzer.process_upload',
            args:{
                filename: frm.doc.upload_imageimage_pdf
            },
            callback: function(response) {
                if (response.message) {
                    // console.log(response.message)
                    frm.doc.output = "<div class=\"d-flex justify-content-center\"><div style=\"display: flex; justify-content: flex-start;\"><div class=\"card\"> <div class=\"card-body\"><p style=\"width: 100%;\">"+
                                     response.message +
                                     "</div></div>";
                    // frm.doc.output = response.message;
                    frm.set_df_property('output','options',frm.doc.output);
                    frm.refresh_field('output')
                    frappe.call({method:'xerebra_ocr.xerebra_ocr.doctype.xerebra_document_analyzer.xerebra_document_analyzer.get_quota_display',
                        args:{
                        },
                        callback: function(response) {
                            if (response.message) {
                                console.log(response.message)
                                frm.doc.usage = response.message
                                frm.refresh_field('usage')
                            }
                        }
                    })
                }
                frm.doc.upload_imageimage_pdf = ''
                frm.refresh_field('upload_imageimage_pdf')
                frappe.validated = false;
            }
        })
    }
});

