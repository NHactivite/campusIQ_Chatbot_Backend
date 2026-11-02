import pymupdf
doc = pymupdf.open()
p1 = doc.new_page()
p1.insert_textbox(pymupdf.Rect(50, 50, 545, 780),
"SAMPLE NOTICE (TEST DATA ONLY)\n\nExamination Form Submission\n\n"
"The last date for submitting the examination form without late fee is 15 December 2026.\n"
"With a late fee of Rs. 500, forms are accepted until 22 December 2026.\n"
"Forms must be submitted through the university examination portal.", fontsize=12)
p2 = doc.new_page()
p2.insert_textbox(pymupdf.Rect(50, 50, 545, 780),
"Scholarship Circular (TEST DATA ONLY)\n\n"
"Students with a family income below Rs. 2,50,000 per year may apply for the merit scholarship.\n"
"Applications close on 10 January 2027 and must include an income certificate.", fontsize=12)
doc.save("docs/sample_notice.pdf")
print("created docs/sample_notice.pdf")
