import timeit
import types
from functools import lru_cache

from pyratemp import Template as PyraTemplate

data = """
#{['user1', 'user2', 'user3']}
    <li>!{element}</li>
    #{['user1', 'user2', 'user3']}
        <li>!{element}</li>
    #{end}
#{end}
?{admin}
    <h1>Hello Admin!</h1>
    <p>Welcome to the admin panel. Here are the users:</p>
    <ul>
        #{['user1', 'user2', 'user3']}
            <li>!{element}</li>
        #{end}
    </ul>
?{end}
?{not admin}
    <h1>Hello User!</h1>
    <p>Unauthorized.</p>
?{end}
?{not admin}
    <h1>Hello User!{1+2}</h1>
    <p>Unauthorized.</p>
?{end}
"""

data_pyratemp = """
<!--(if admin)-->
    <h1>Hello Admin!</h1>
    <p>Welcome to the admin panel. Here are the users:</p>
    <ul>
        <!--(for user in ['user1', 'user2', 'user3'])-->
            <li>$! user !$</li>
        <!--(end)-->
    </ul>

<!--(else)-->
    <h1>Hello User!</h1>
    <p>Unauthorized.</p>
<!--(end)-->
"""


class SandboxedEnvironment:
    def __init__(self):

        self.__external_modules = {
            "random": self.__import("random"),
            "math": self.__import("math")
        }

        self.__blacklist = [
            # strings
            "format",

            # unsafe functions
            "system",
            "open",
            "popen",
            "exec",
            "compile",
            "eval",
            "__import__",
            "breakpoint",

            # unsafe attributes
            "__builtins__",
            "__globals__",

            # unsafe modules
            "sys",
            "os",
            "nt",
            "subprocess",
            "shutil",
            "pickle",
            "marshal",
            "ctypes",
            "importlib"
        ]

        self.__globals = {
            # Other
            "false": False,
            "null": None,
            "true": True,

            # Useful functions
            "abs": abs,
            "all": all,
            "any": any,
            "ascii": ascii,
            "bin": bin,
            "chr": chr,
            "divmod": divmod,
            "format": format,
            "hasattr": hasattr,
            "hex": hex,
            "isinstance": isinstance,
            "iter": iter,
            "len": len,
            "max": max,
            "min": min,
            "next": next,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "repr": repr,
            "round": round,
            "sorted": sorted,
            "sum": sum,

            # Safe types
            "bool": bool,
            "bytearray": bytearray,
            "bytes": bytes,
            "complex": complex,
            "dict": dict,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "frozenset": frozenset,
            "int": int,
            "reversed": reversed,
            "range": range,
            "map": map,
            "tuple": tuple,
            "zip": zip,
            "list": list,
            "set": set,
            "str": str,
            "type": type,

            # Other
            "randstr": self.__ext_randstr,
            "randint": self.__external_modules["random"].randint,
            "randfloat": self.__external_modules["random"].uniform,
            "randbits": self.__external_modules["random"].getrandbits,

            # Math
            "sqrt": self.__external_modules["math"].sqrt,
            "pi": self.__external_modules["math"].pi,
            "euler_e": self.__external_modules["math"].e,
            "sin": self.__external_modules["math"].sin,
            "cos": self.__external_modules["math"].cos
        }

    @lru_cache
    def __import(self, module_name):
        return __import__(module_name)

    def __ext_randstr(self, l, charset="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
        return ''.join(self.__external_modules["random"].choice(charset) for _ in range(l))

    @lru_cache
    def __check_blacklist(self, k):
        r = f"{getattr(k, "__module__", getattr(k, "__name__", k))}"
        if r in self.__blacklist or r.startswith("_"):
            raise ValueError(f"Unsafe entity: {k!r}")

    @lru_cache
    def __compile_and_check(self, cobj: types.CodeType):
        for k in cobj.co_names + cobj.co_varnames + \
            cobj.co_consts + cobj.co_freevars + cobj.co_cellvars:

            if isinstance(k, types.CodeType):
                self.__compile_and_check(k)

            self.__check_blacklist(k)
        
        return cobj
    
    @lru_cache
    def __compile(self, code):
        return self.__compile_and_check(compile(code, "<string>", "eval"))

    def get_result(self, code, eval_kwargs, raw=False): # TODO: Distinct between globals and locals        
        compiled = self.__compile(code)
        result = eval(compiled, self.__globals, eval_kwargs)

        return result if raw else f'{result}'

class Template:

    @staticmethod
    def measure_time(func):
        total_time = 0

        def wrapper(*args, **kwargs):
            nonlocal total_time
            start_time = timeit.default_timer()
            result = func(*args, **kwargs)
            end_time = timeit.default_timer()
            total_time += end_time - start_time
            return result

        def get_total_time():
            return total_time

        wrapper.get_total_time = get_total_time
        wrapper.original_function = func

        return wrapper

    @measure_time
    def __init__(self, template):
        self.template = template
        self.__loop_counter = 0
        self.__sandboxed_environment = SandboxedEnvironment()
    
    @measure_time
    def render(self, **kwargs):
        return self.__render(self.template, **kwargs)


    @staticmethod
    @lru_cache
    @measure_time
    def calculate_ptr_value(ptr: int, res: str, start_idx: int, end_idx: int) -> int:
        return ptr + len(res) - (end_idx - start_idx + 1)

    @staticmethod
    @lru_cache
    @measure_time
    def sub(template: str, replacement: str, start_idx: int, end_idx: int) -> str:
        return template[:start_idx] + replacement + template[end_idx+1:]

    @staticmethod
    @lru_cache
    @measure_time
    def get_next_external_block(string, delim): # We want to exclude the nested for loops
        stack = []
        for x in range(len(string) - 5):
            if string[x:x+6] == delim + '{end}':
                
                to_ret = (stack.pop(), x + 6)
                if not stack:
                    return to_ret

            elif string[x] == delim and string[x:x+6] != delim + '{end}':
                stack.append(x)
        
        raise ValueError(f"Missing {delim}{{end}}")


    @lru_cache
    @measure_time
    def parse_inline(self, ptr: int, template: str):
        start_idx = ptr + 1
        end_idx = ptr + 1

        done_pre = False
        
        while not done_pre and end_idx > ptr: # TODO: Handle curly brackets inside strings
            if template[end_idx] == '}':
                if template[end_idx - 1] != '\\':
                    done_pre = True
                    to_return = template[start_idx + 1:end_idx]
                    if not to_return:
                        raise ValueError(f"Template string cannot be null at line {self.template.count('\n', 0, start_idx)}")
                    return to_return, end_idx, start_idx, end_idx


            end_idx += 1

    @lru_cache
    @measure_time
    def parse_block(self, block_data, ptr, start_from=0, delim="#"):
        to_parse = block_data[ptr:]
        
        block = self.get_next_external_block(to_parse, delim)

        start, end = block
    
        data_block = to_parse[start:end]
        
        to_eval, *_, end_idx = self.parse_inline(start_from, data_block)
        data_to_render = data_block[end_idx + 1:-6]


        return start, end, to_eval, data_block, data_to_render
    
    @lru_cache
    @measure_time
    def __render(self, template: str, **kwargs: dict):
        ptr = 0

        while ptr < len(template):
            # Decomment this to make it interactive
            # print(self.sub(template, "X", ptr, ptr), end="\r")
            # import os
            
            # import time

            # input()
            # os.system("cls")
            match template[ptr]:
                case '#':
                    if template[max(0, ptr - 1)] != '\\' and template[ptr + 1:ptr + 6] != '{end}':
                        is_static = template[ptr + 1].lower() == 'r'

                        start, end, to_eval, data_loop, data_to_render = self.parse_block(template, ptr, start_from=is_static, delim="#")

                        res = self.__sandboxed_environment.get_result(to_eval, kwargs, raw=True)

                        to_sub = ""
                        loc = locals()

                        self.__loop_counter += 1
                        for loc[f"index_{self.__loop_counter}"], element in enumerate(res):
                            if not is_static: # is_static == True ? Enable caching
                                kwargs[f"index_{self.__loop_counter}"] = loc[f"index_{self.__loop_counter}"]
                                kwargs["element"] = element
                                    
                            to_sub += self.__render(data_to_render, **kwargs).rstrip()
                        
                        del loc[f"index_{self.__loop_counter}"]
                        self.__loop_counter -= 1

                        data_loop = self.sub(data_loop, to_sub, 0, end + 6)
                        template = self.sub(template, data_loop, ptr + start, ptr + end - 1)
                        ptr += len(data_loop) - 1

                case '?':
                    if template[max(0, ptr - 1)] != '\\' and template[ptr + 1:ptr + 6] != '{end}':

                        start, end, to_eval, data_if_block, data_to_render = self.parse_block(template, ptr, start_from=0, delim="?")
                        res = self.__sandboxed_environment.get_result(to_eval, kwargs, raw=True)

                        to_sub = ""
                        if res: to_sub += self.__render(data_to_render, **kwargs).rstrip()

                        data_if_block = self.sub(data_if_block, to_sub, 0, end + 6)
                        template = self.sub(template, data_if_block, ptr + start, ptr + end - 1)
                        ptr += len(data_if_block) - 1

                case '!':
                    if template[ptr + 1] != '{':
                        ptr += 1
                        continue

                    if template[max(0, ptr - 1)] == '\\' and template[ptr + 1] == '{':
                        template = self.sub(template, '!', ptr - 1, ptr)
                        continue

                    
                    to_eval, ptr, start_idx, end_idx = self.parse_inline(ptr, template)

                    res = self.__sandboxed_environment.get_result(to_eval, kwargs)
                    ptr = self.calculate_ptr_value(ptr, res, start_idx, end_idx) - 1

                    template = self.sub(template, res, start_idx - 1, end_idx)

            ptr += 1

        return template



def test_pyratemp():
    return PyraTemplate(data_pyratemp)(admin=False)

def test_eztemplate():
    return Template(data).render(admin=False)


print(Template(data).render(admin=False))
# print(timeit.timeit(test_eztemplate, number=1000))
# print(timeit.timeit(test_pyratemp, number=1000))
# print("OO")

# # print(Template(data).render())


# print(Template.parse_inline.get_total_time())
# print(Template.get_next_external_block.get_total_time())

# print(Template(data).render(admin=False))
# for x in dir(Template):
#     r = getattr(Template, x)
#     if hasattr(r, "get_total_time"):
#         print(f"{x}: {r.get_total_time()}")