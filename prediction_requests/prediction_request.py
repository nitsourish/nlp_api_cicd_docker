from marshmallow import Schema
from flask_marshmallow.fields import fields


class PredictionRequest(Schema):
    text = fields.String(required=True)
