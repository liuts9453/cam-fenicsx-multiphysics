import os
import ast
import sys

def get_classes_in_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=filepath)
    return [node.name for node in tree.body if isinstance(node, ast.ClassDef)]

def generate_init_py_recursively(base_dir):
    for dirpath, _, filenames in os.walk(base_dir):
        import_lines = []
        for filename in filenames:
            if filename.endswith(".py") and filename != "__init__.py":
                filepath = os.path.join(dirpath, filename)
                module_name = os.path.splitext(filename)[0]
                class_names = get_classes_in_file(filepath)
                if class_names:
                    imports = ", ".join(class_names)
                    import_lines.append(f"from .{module_name} import {imports}")

        if import_lines:
            init_path = os.path.join(dirpath, "__init__.py")
            with open(init_path, "w", encoding="utf-8") as f:
                f.write("# Auto-generated __init__.py\n")
                f.write("\n".join(import_lines))
            print(f"[✓] Generated {init_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_init.py <target_directory>")
        sys.exit(1)

    target_dir = sys.argv[1]
    if not os.path.isdir(target_dir):
        print(f"[✗] Directory not found: {target_dir}")
        sys.exit(1)

    generate_init_py_recursively(target_dir)

