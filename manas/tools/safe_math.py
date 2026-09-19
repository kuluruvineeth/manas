import ast
import math
import operator

CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
FUNCTION_NAMES = [
    "sqrt", "exp", "log", "log2", "log10",
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh",
    "floor", "ceil", "trunc", "fabs", "fmod", "hypot", "gcd", "degrees", "radians",
]
UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
SUBSTITUTIONS = str.maketrans({"^": "**", "×": "*", "÷": "/", "−": "-", "²": "**2", "³": "**3", "（": "(", "）": ")"})
MAX_EXPRESSION_CHARS = 512


def guarded_pow(base, exponent):
    too_large = abs(exponent) > 1e4 or (abs(base) > 1 and abs(exponent) * math.log10(abs(base)) > 100)
    if too_large:
        raise ValueError("power result too large")
    return base**exponent


FUNCTIONS = {"pow": guarded_pow, **{name: getattr(math, name) for name in FUNCTION_NAMES}}
BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: guarded_pow,
}


def _name_of(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and getattr(node.value, "id", "") == "math":
        return node.attr
    return None


def _walk(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    name = _name_of(node)
    if name in CONSTANTS:
        return CONSTANTS[name]
    if isinstance(node, ast.UnaryOp):
        return UNARY_OPS[type(node.op)](_walk(node.operand))
    if isinstance(node, ast.BinOp):
        return BINARY_OPS[type(node.op)](_walk(node.left), _walk(node.right))
    if isinstance(node, ast.Call) and not node.keywords:
        return FUNCTIONS[_name_of(node.func)](*map(_walk, node.args))
    raise ValueError(f"unsupported expression syntax: {type(node).__name__}")


def safe_math_eval(expression):
    text = str(expression).translate(SUBSTITUTIONS).strip()
    if not text or len(text) > MAX_EXPRESSION_CHARS:
        raise ValueError("expression is empty or too long")
    try:
        return _walk(ast.parse(text, mode="eval").body)
    except (KeyError, SyntaxError, TypeError) as error:
        raise ValueError("unsupported expression syntax") from error
