# Groovygo

A minimal **Groovy-like scripting language** implemented in **Go**. It features dynamic typing, functions, maps, arrays, dot-call access, and a simple REPL.

---

## 🚀 Features

* Dynamic types: `number`, `string`, `bool`, `nil`, arrays, maps
* First-class functions with closures
* Dot-style property access (e.g., `obj.key`)
* Simple built-in functions (`println`, `len`)
* REPL + script file execution
* Easy to extend (lexer, parser, AST, evaluator)

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourname/groovygo.git
cd groovygo
```

Build:

```bash
go build
```

Run REPL:

```bash
./groovygo
```

Run a script:

```bash
./groovygo example.groovygo
```

---

## 📜 Example Script (`example.groovygo`)

```groovy
println("Hello from groovygo!")

x = 10
y = 32
println("x + y = " + (x + y))

nums = [1, 2, 3]
nums.append(4)
println(nums)

person = {
  name: "Alice",
  age: 30
}
println(person.name)

square = fn(n) {
  return n * n
}
println(square(5))
```

---

## 🛠 Project Structure

```
/lexer        – token definitions + lexical scanner
/parser       – Pratt parser + AST
/object       – runtime object system
/evaluator    – evaluator for AST nodes
main.go       – REPL and script runner
```

---

## 🔧 Extending the Language

You can easily add:

* Classes / prototypes
* Modules
* Standard library functions
* Bytecode + VM backend

---

## 📄 License

MIT License

---

## 🤝 Contributing

PRs are welcome!

If you want a full GitHub repository layout (folders + files), just ask!
