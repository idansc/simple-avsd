#!/usr/bin/python
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import sys

def send_mail(model_name, model, msg_txt):
    # Send mail
    # -------------
    # Import smtplib for the actual sending function
    import smtplib

    # Import the email modules we'll need
    from email.mime.text import MIMEText
    import socket

    # Create a text/plain message
    msg = MIMEText(msg_txt)

    msg['Subject'] = "Exp: " + model_name + " " + socket.gethostname() + " " + model +"\n"

    # me == the sender's email address
    # family = the list of all recipients' email addresses
    family = ["zeev0595@gmail.com", "alexander.g.schwing@gmail.com", "tamir.hazan@gmail.com"]
    msg['From'] = 'idansc@tx.technion.ac.il'
    msg['To'] = ', '.join(family)
    msg.preamble = 'AudioDial'

    # Open the files in binary mode.  Use imghdr to figure out the
    # MIME subtype for each specific image.

    # with open(os.path.join(mydir, "plot.png"), 'rb') as fp:
    #    img_data = fp.read()
    #    msg.add_attachment(img_data, maintype='image',
    #                       subtype=imghdr.what(None, img_data))

    # Send the email via our own SMTP server.
    s = smtplib.SMTP('tx.technion.ac.il')
    s.sendmail('idansc@tx.technion.ac.il', family, msg.as_string())
    s.quit()


# create coco object and cocoRes object
coco = COCO(sys.argv[1])
cocoRes = coco.loadRes(sys.argv[2])
model_name = sys.argv[3]
model = sys.argv[4]

# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)
# evaluate on a subset of images by setting
cocoEval.params['image_id'] = cocoRes.getImgIds()

# evaluate results
cocoEval.evaluate()
# print output evaluation scores
email_msg = ""
for metric, score in cocoEval.eval.items():
    print '%s: %.3f' % (metric, score)
    email_msg += '%s: %.3f\n' % (metric, score)
for key, value in cocoEval.imgToEval.iteritems():
    print key, value

send_mail(model_name, model, email_msg)
