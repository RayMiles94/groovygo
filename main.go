// groovygo: Minimal Groovy-like scripting language written in Go
// -----------------------------------------------------------------
// Features included:
//   - Dynamic typing (all values are runtime values)
//   - Basic types: nil, bool, number (float64), string
//   - Arrays (list), Maps (associative)
//   - Variables with lexical scope
//   - Functions (first-class, closures)
//   - Method-like calls for arrays/maps/strings (dot call semantics)
//   - Simple 'class' style objects using map + prototype (very lightweight)
//   - REPL and example script execution
//
// This is intentionally small to be a base for experimentation.
// To run:
//
//	go run main.go
//
// or
//
//	go build -o groovygo && ./groovygo script.groovy
//
// Example language snippets (try these in REPL or put them in script.groovy):
//
// x = 10
// y = 20
// println(x + y)
//
// def add = fn(a, b) { return a + b }
// println(add(3,4))
//
// list = [1,2,3]
// list.append(4)
// println(list)
//
// map = { "name": "Ada", "age": 37 }
// println(map.name)
// map.set("city", "London")
// println(map)
//
// # closure
// def makeAdder = fn(n) { fn(x) { x + n } }
// incBy5 = makeAdder(5)
// println(incBy5(10))
//
// -----------------------------------------------------------------
package main

import (
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"unicode"
)

// ---------- Lexer ----------

type TokenType string

const (
	TOK_ILLEGAL TokenType = "ILLEGAL"
	TOK_EOF     TokenType = "EOF"

	TOK_IDENT  TokenType = "IDENT"
	TOK_NUMBER TokenType = "NUMBER"
	TOK_STRING TokenType = "STRING"

	TOK_ASSIGN TokenType = "="
	TOK_PLUS   TokenType = "+"
	TOK_MINUS  TokenType = "-"
	TOK_BANG   TokenType = "!"
	TOK_ASTER  TokenType = "*"
	TOK_SLASH  TokenType = "/"
	TOK_LT     TokenType = "<"
	TOK_GT     TokenType = ">"

	TOK_COMMA  TokenType = ","
	TOK_SEMI   TokenType = ";"
	TOK_COLON  TokenType = ":"
	TOK_LPAREN TokenType = "("
	TOK_RPAREN TokenType = ")"
	TOK_LBRACE TokenType = "{"
	TOK_RBRACE TokenType = "}"
	TOK_LBRACK TokenType = "["
	TOK_RBRACK TokenType = "]"
	TOK_DOT    TokenType = "."

	TOK_FUNCTION TokenType = "FN"
	TOK_LET      TokenType = "LET"
	TOK_RETURN   TokenType = "RETURN"
)

var keywords = map[string]TokenType{
	"fn":     TOK_FUNCTION,
	"def":    TOK_LET,
	"return": TOK_RETURN,
	"true":   TOK_IDENT,
	"false":  TOK_IDENT,
	"nil":    TOK_IDENT,
}

type Token struct {
	Type    TokenType
	Literal string
}

func lookupIdent(ident string) TokenType {
	if tok, ok := keywords[ident]; ok {
		return tok
	}
	return TOK_IDENT
}

type Lexer struct {
	input        string
	position     int
	readPosition int
	ch           byte
}

func NewLexer(input string) *Lexer {
	l := &Lexer{input: input}
	l.readChar()
	return l
}

func (l *Lexer) readChar() {
	if l.readPosition >= len(l.input) {
		l.ch = 0
	} else {
		l.ch = l.input[l.readPosition]
	}
	l.position = l.readPosition
	l.readPosition++
}

func (l *Lexer) peekChar() byte {
	if l.readPosition >= len(l.input) {
		return 0
	}
	return l.input[l.readPosition]
}

func (l *Lexer) NextToken() Token {
	l.skipWhitespace()
	var tok Token

	switch l.ch {
	case '=':
		tok = newToken(TOK_ASSIGN, l.ch)
	case '+':
		tok = newToken(TOK_PLUS, l.ch)
	case '-':
		tok = newToken(TOK_MINUS, l.ch)
	case '!':
		tok = newToken(TOK_BANG, l.ch)
	case '*':
		tok = newToken(TOK_ASTER, l.ch)
	case '/':
		tok = newToken(TOK_SLASH, l.ch)
	case '<':
		tok = newToken(TOK_LT, l.ch)
	case '>':
		tok = newToken(TOK_GT, l.ch)
	case ',':
		tok = newToken(TOK_COMMA, l.ch)
	case ';':
		tok = newToken(TOK_SEMI, l.ch)
	case ':':
		tok = newToken(TOK_COLON, l.ch)
	case '(':
		tok = newToken(TOK_LPAREN, l.ch)
	case ')':
		tok = newToken(TOK_RPAREN, l.ch)
	case '{':
		tok = newToken(TOK_LBRACE, l.ch)
	case '}':
		tok = newToken(TOK_RBRACE, l.ch)
	case '[':
		tok = newToken(TOK_LBRACK, l.ch)
	case ']':
		tok = newToken(TOK_RBRACK, l.ch)
	case '.':
		tok = newToken(TOK_DOT, l.ch)
	case '"':
		tok.Type = TOK_STRING
		tok.Literal = l.readString()
		l.readChar()
		return tok
	case 0:
		tok.Literal = ""
		tok.Type = TOK_EOF
		return tok
	default:
		if isLetter(l.ch) {
			lit := l.readIdentifier()
			tok.Type = lookupIdent(lit)
			tok.Literal = lit
			return tok
		} else if isDigit(l.ch) {
			lit := l.readNumber()
			tok.Type = TOK_NUMBER
			tok.Literal = lit
			return tok
		} else {
			tok = newToken(TOK_ILLEGAL, l.ch)
		}
	}
	l.readChar()
	return tok
}

func newToken(t TokenType, ch byte) Token {
	return Token{Type: t, Literal: string(ch)}
}

func (l *Lexer) skipWhitespace() {
	for l.ch == ' ' || l.ch == '\t' || l.ch == '\n' || l.ch == '\r' {
		l.readChar()
	}
}

func isLetter(ch byte) bool {
	return unicode.IsLetter(rune(ch)) || ch == '_' || ch == '$'
}

func (l *Lexer) readIdentifier() string {
	pos := l.position
	for isLetter(l.ch) || isDigit(l.ch) {
		l.readChar()
	}
	return l.input[pos:l.position]
}

func isDigit(ch byte) bool {
	return '0' <= ch && ch <= '9'
}

func (l *Lexer) readNumber() string {
	pos := l.position
	for isDigit(l.ch) || l.ch == '.' {
		l.readChar()
	}
	return l.input[pos:l.position]
}

func (l *Lexer) readString() string {
	// l.ch == '"' currently
	l.readChar()
	pos := l.position
	for l.ch != '"' && l.ch != 0 {
		l.readChar()
	}
	return l.input[pos:l.position]
}

// ---------- Parser (very small expression grammar) ----------

// We'll implement a Pratt parser for expressions and simple statements

type Node interface{}

type Program struct {
	Statements []Node
}

// Statements

type ExpressionStatement struct {
	Expression Node
}

type ReturnStatement struct {
	Value Node
}

type LetStatement struct {
	Name  string
	Value Node
}

// Expressions

type Identifier struct{ Value string }

type NumberLiteral struct{ Value float64 }

type StringLiteral struct{ Value string }

type BooleanLiteral struct{ Value bool }

type NilLiteral struct{}

type PrefixExpression struct {
	Operator string
	Right    Node
}

type InfixExpression struct {
	Left     Node
	Operator string
	Right    Node
}

type IfExpression struct {
	Condition   Node
	Consequence *BlockStatement
	Alternative *BlockStatement
}

type BlockStatement struct {
	Statements []Node
}

type FunctionLiteral struct {
	Parameters []string
	Body       *BlockStatement
}

type CallExpression struct {
	Function  Node
	Arguments []Node
}

type ArrayLiteral struct{ Elements []Node }

type IndexExpression struct {
	Left  Node
	Index Node
}

type HashLiteral struct{ Pairs map[string]Node }

type DotExpression struct {
	Left       Node
	Identifier string
	Args       []Node
}

// parser not fully featured, but enough for demo

type Parser struct {
	l    *Lexer
	cur  Token
	peek Token
}

func NewParser(l *Lexer) *Parser {
	p := &Parser{l: l}
	p.nextToken()
	p.nextToken()
	return p
}

func (p *Parser) nextToken() {
	p.cur = p.peek
	p.peek = p.l.NextToken()
}

func (p *Parser) ParseProgram() *Program {
	prog := &Program{}
	for p.cur.Type != TOK_EOF {
		stmt := p.parseStatement()
		if stmt != nil {
			prog.Statements = append(prog.Statements, stmt)
		}
		p.nextToken()
	}
	return prog
}

func (p *Parser) parseStatement() Node {
	switch p.cur.Type {
	case TOK_RETURN:
		p.nextToken()
		expr := p.parseExpression(0)
		return &ReturnStatement{Value: expr}
	case TOK_LET:
		// def <ident> = <expr>
		p.nextToken()
		if p.cur.Type != TOK_IDENT {
			return nil
		}
		name := p.cur.Literal
		p.nextToken() // go to =
		p.nextToken()
		val := p.parseExpression(0)
		return &LetStatement{Name: name, Value: val}
	default:
		expr := p.parseExpression(0)
		return &ExpressionStatement{Expression: expr}
	}
}

// precedence
const (
	_ int = iota
	LOWEST
	SUM     // + -
	PRODUCT // * /
	PREFIX  // -X !X
	CALL    // myfunc(X)
)

func (p *Parser) tokenPrecedence(t TokenType) int {
	switch t {
	case TOK_PLUS, TOK_MINUS:
		return SUM
	case TOK_ASTER, TOK_SLASH:
		return PRODUCT
	case TOK_LPAREN, TOK_LBRACK, TOK_DOT:
		return CALL
	}
	return LOWEST
}

func (p *Parser) parseExpression(precedence int) Node {
	var left Node

	// prefix
	switch p.cur.Type {
	// removed duplicate TOK_IDENT case
	//	left = &Identifier{Value: p.cur.Literal}
	case TOK_NUMBER:
		v, _ := strconv.ParseFloat(p.cur.Literal, 64)
		left = &NumberLiteral{Value: v}
	case TOK_STRING:
		left = &StringLiteral{Value: p.cur.Literal}
	case TOK_BANG, TOK_MINUS:
		op := p.cur.Literal
		p.nextToken()
		r := p.parseExpression(PREFIX)
		left = &PrefixExpression{Operator: op, Right: r}
	case TOK_LPAREN:
		p.nextToken()
		left = p.parseExpression(0)
		// expect )
	case TOK_LBRACK:
		// array literal
		p.nextToken()
		var elems []Node
		for p.cur.Type != TOK_RBRACK && p.cur.Type != TOK_EOF {
			el := p.parseExpression(0)
			elems = append(elems, el)
			if p.peek.Type == TOK_COMMA {
				p.nextToken()
				p.nextToken()
			}
		}
		left = &ArrayLiteral{Elements: elems}
	case TOK_LBRACE:
		// simple hash literal: { "k": v }
		p.nextToken()
		pairs := map[string]Node{}
		for p.cur.Type != TOK_RBRACE && p.cur.Type != TOK_EOF {
			if p.cur.Type == TOK_STRING {
				k := p.cur.Literal
				p.nextToken() // should be :
				p.nextToken()
				v := p.parseExpression(0)
				pairs[k] = v
				if p.peek.Type == TOK_COMMA {
					p.nextToken()
					p.nextToken()
				}
			}
		}
		left = &HashLiteral{Pairs: pairs}
	case TOK_FUNCTION:
		// fn (params) { body }
		p.nextToken()
		// expect (
		p.nextToken() // skip to maybe lp
		params := []string{}
		if p.cur.Type == TOK_LPAREN {
			p.nextToken()
			for p.cur.Type != TOK_RPAREN && p.cur.Type != TOK_EOF {
				if p.cur.Type == TOK_IDENT {
					params = append(params, p.cur.Literal)
				}
				p.nextToken()
				if p.cur.Type == TOK_COMMA {
					p.nextToken()
				}
			}
			p.nextToken()
		}
		// body
		if p.cur.Type == TOK_LBRACE {
			body := p.parseBlock()
			left = &FunctionLiteral{Parameters: params, Body: body}
		}
	case TOK_IDENT:
		left = &Identifier{Value: p.cur.Literal}
	case TOK_ILLEGAL:
		left = nil
	case TOK_EOF:
		left = nil
	default:
		left = nil
	}

	// infix / call parsing
	for p.peek.Type != TOK_SEMI && precedence < p.tokenPrecedence(p.peek.Type) && p.peek.Type != TOK_EOF {
		p.nextToken()
		switch p.cur.Type {
		case TOK_PLUS, TOK_MINUS, TOK_ASTER, TOK_SLASH:
			op := p.cur.Literal
			p.nextToken()
			right := p.parseExpression(p.tokenPrecedence(TOK_PLUS))
			left = &InfixExpression{Left: left, Operator: op, Right: right}
		case TOK_LPAREN:
			// call
			// parse arguments
			args := []Node{}
			p.nextToken()
			for p.cur.Type != TOK_RPAREN && p.cur.Type != TOK_EOF {
				arg := p.parseExpression(0)
				args = append(args, arg)
				if p.peek.Type == TOK_COMMA {
					p.nextToken()
					p.nextToken()
				}
			}
			left = &CallExpression{Function: left, Arguments: args}
		case TOK_DOT:
			// dot expression: <left> . ident (args?)
			p.nextToken()
			id := p.cur.Literal
			args := []Node{}
			if p.peek.Type == TOK_LPAREN {
				p.nextToken()
				p.nextToken()
				for p.cur.Type != TOK_RPAREN && p.cur.Type != TOK_EOF {
					arg := p.parseExpression(0)
					args = append(args, arg)
					if p.peek.Type == TOK_COMMA {
						p.nextToken()
						p.nextToken()
					}
				}
			}
			left = &DotExpression{Left: left, Identifier: id, Args: args}
		}
	}

	return left
}

func (p *Parser) parseBlock() *BlockStatement {
	bs := &BlockStatement{}
	p.nextToken()
	for p.cur.Type != TOK_RBRACE && p.cur.Type != TOK_EOF {
		stmt := p.parseStatement()
		if stmt != nil {
			bs.Statements = append(bs.Statements, stmt)
		}
		p.nextToken()
	}
	return bs
}

// ---------- Evaluator ----------

type ObjectType string

const (
	OBJ_NIL    ObjectType = "NIL"
	OBJ_NUMBER ObjectType = "NUMBER"
	OBJ_STRING ObjectType = "STRING"
	OBJ_BOOL   ObjectType = "BOOL"
	OBJ_ARRAY  ObjectType = "ARRAY"
	OBJ_MAP    ObjectType = "MAP"
	OBJ_FN     ObjectType = "FUNCTION"
)

type Object interface {
	Type() ObjectType
	Inspect() string
}

type Nil struct{}

func (n *Nil) Type() ObjectType { return OBJ_NIL }
func (n *Nil) Inspect() string  { return "nil" }

type Number struct{ Value float64 }

func (n *Number) Type() ObjectType { return OBJ_NUMBER }
func (n *Number) Inspect() string  { return fmt.Sprintf("%v", n.Value) }

type Str struct{ Value string }

func (s *Str) Type() ObjectType { return OBJ_STRING }
func (s *Str) Inspect() string  { return s.Value }

type Bool struct{ Value bool }

func (b *Bool) Type() ObjectType { return OBJ_BOOL }
func (b *Bool) Inspect() string  { return fmt.Sprintf("%v", b.Value) }

type Array struct{ Elements []Object }

func (a *Array) Type() ObjectType { return OBJ_ARRAY }
func (a *Array) Inspect() string {
	parts := []string{}
	for _, e := range a.Elements {
		parts = append(parts, e.Inspect())
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

type MapObj struct{ Pairs map[string]Object }

func (m *MapObj) Type() ObjectType { return OBJ_MAP }
func (m *MapObj) Inspect() string {
	parts := []string{}
	for k, v := range m.Pairs {
		parts = append(parts, fmt.Sprintf("%q: %s", k, v.Inspect()))
	}
	return "{" + strings.Join(parts, ", ") + "}"
}

type Function struct {
	Params []string
	Body   *BlockStatement
	Env    *Environment
}

func (f *Function) Type() ObjectType { return OBJ_FN }
func (f *Function) Inspect() string  { return fmt.Sprintf("fn(%v) {...}", strings.Join(f.Params, ",")) }

// Environment

type Environment struct {
	store map[string]Object
	outer *Environment
}

func NewEnvironment() *Environment                { return &Environment{store: map[string]Object{}} }
func NewEnclosed(outer *Environment) *Environment { e := NewEnvironment(); e.outer = outer; return e }

func (e *Environment) Get(name string) (Object, bool) {
	obj, ok := e.store[name]
	if !ok && e.outer != nil {
		return e.outer.Get(name)
	}
	return obj, ok
}

func (e *Environment) Set(name string, val Object) Object {
	e.store[name] = val
	return val
}

// Eval

func Eval(node Node, env *Environment) Object {
	switch n := node.(type) {
	case *Program:
		var result Object = &Nil{}
		for _, s := range n.Statements {
			result = Eval(s, env)
			if ret, ok := result.(*ReturnValue); ok {
				return ret.Value
			}
		}
		return result
	case *ExpressionStatement:
		return Eval(n.Expression, env)
	case *NumberLiteral:
		return &Number{Value: n.Value}
	case *StringLiteral:
		return &Str{Value: n.Value}
	case *Identifier:
		// true/false/nil
		if n.Value == "true" {
			return &Bool{Value: true}
		}
		if n.Value == "false" {
			return &Bool{Value: false}
		}
		if n.Value == "nil" {
			return &Nil{}
		}
		if val, ok := env.Get(n.Value); ok {
			return val
		}
		return &Nil{}
	case *LetStatement:
		val := Eval(n.Value, env)
		return env.Set(n.Name, val)
	case *ReturnStatement:
		v := Eval(n.Value, env)
		return &ReturnValue{Value: v}
	case *PrefixExpression:
		r := Eval(n.Right, env)
		return evalPrefixExpression(n.Operator, r)
	case *InfixExpression:
		left := Eval(n.Left, env)
		right := Eval(n.Right, env)
		return evalInfixExpression(n.Operator, left, right)
	case *ArrayLiteral:
		els := []Object{}
		for _, e := range n.Elements {
			els = append(els, Eval(e, env))
		}
		return &Array{Elements: els}
	case *HashLiteral:
		pairs := map[string]Object{}
		for k, v := range n.Pairs {
			pairs[k] = Eval(v, env)
		}
		return &MapObj{Pairs: pairs}
	case *FunctionLiteral:
		return &Function{Params: n.Parameters, Body: n.Body, Env: env}
	case *CallExpression:
		fn := Eval(n.Function, env)
		args := []Object{}
		for _, a := range n.Arguments {
			args = append(args, Eval(a, env))
		}
		return applyFunction(fn, args)
	case *BlockStatement:
		var res Object = &Nil{}
		for _, s := range n.Statements {
			res = Eval(s, env)
			if ret, ok := res.(*ReturnValue); ok {
				return ret
			}
		}
		return res
	case *DotExpression:
		left := Eval(n.Left, env)
		// support built-in methods for arrays, maps, strings, numbers
		return evalDot(left, n.Identifier, n.Args, env)
	}
	return &Nil{}
}

// Return wrapper

type ReturnValue struct{ Value Object }

func (rv *ReturnValue) Type() ObjectType { return "RETURN" }
func (rv *ReturnValue) Inspect() string  { return rv.Value.Inspect() }

func evalPrefixExpression(op string, right Object) Object {
	switch op {
	case "-":
		if rn, ok := right.(*Number); ok {
			return &Number{Value: -rn.Value}
		}
	case "!":
		return &Bool{Value: !isTruthy(right)}
	}
	return &Nil{}
}

func isTruthy(obj Object) bool {
	switch o := obj.(type) {
	case *Nil:
		return false
	case *Bool:
		return o.Value
	default:
		return true
	}
}

func evalInfixExpression(op string, left, right Object) Object {
	// number operations
	ln, lok := left.(*Number)
	rn, rok := right.(*Number)
	if lok && rok {
		switch op {
		case "+":
			return &Number{Value: ln.Value + rn.Value}
		case "-":
			return &Number{Value: ln.Value - rn.Value}
		case "*":
			return &Number{Value: ln.Value * rn.Value}
		case "/":
			return &Number{Value: ln.Value / rn.Value}
		case "<":
			return &Bool{Value: ln.Value < rn.Value}
		case ">":
			return &Bool{Value: ln.Value > rn.Value}
		}
	}
	// string concat
	if lstr, ok := left.(*Str); ok {
		if op == "+" {
			if rstr, rok := right.(*Str); rok {
				return &Str{Value: lstr.Value + rstr.Value}
			}
			// concat with number
			return &Str{Value: lstr.Value + right.Inspect()}
		}
	}
	return &Nil{}
}

func applyFunction(fn Object, args []Object) Object {
	switch f := fn.(type) {
	case *Function:
		env := NewEnclosed(f.Env)
		for i, p := range f.Params {
			if i < len(args) {
				env.Set(p, args[i])
			}
		}
		res := Eval(f.Body, env)
		if rv, ok := res.(*ReturnValue); ok {
			return rv.Value
		}
		return res
	case *Builtin:
		return f.Fn(args...)
	}
	return &Nil{}
}

// Builtins

type Builtin struct{ Fn func(...Object) Object }

func (b *Builtin) Type() ObjectType { return OBJ_FN }
func (b *Builtin) Inspect() string  { return "<builtin>" }

func registerBuiltins(env *Environment) {
	env.Set("println", &Builtin{Fn: func(args ...Object) Object {
		parts := []string{}
		for _, a := range args {
			parts = append(parts, a.Inspect())
		}
		fmt.Println(strings.Join(parts, " "))
		return &Nil{}
	}})

	env.Set("len", &Builtin{Fn: func(args ...Object) Object {
		if len(args) == 0 {
			return &Number{Value: 0}
		}
		switch a := args[0].(type) {
		case *Array:
			return &Number{Value: float64(len(a.Elements))}
		case *Str:
			return &Number{Value: float64(len(a.Value))}
		case *MapObj:
			return &Number{Value: float64(len(a.Pairs))}
		}
		return &Number{Value: 0}
	}})
}

func evalDot(left Object, ident string, argNodes []Node, env *Environment) Object {
	// evaluate args
	args := []Object{}
	for _, n := range argNodes {
		args = append(args, Eval(n, env))
	}
	switch l := left.(type) {
	case *Array:
		if ident == "append" && len(args) == 1 {
			l.Elements = append(l.Elements, args[0])
			return &Nil{}
		}
		if ident == "get" && len(args) == 1 {
			if idx, ok := args[0].(*Number); ok {
				i := int(idx.Value)
				if i >= 0 && i < len(l.Elements) {
					return l.Elements[i]
				}
			}
			return &Nil{}
		}
	case *MapObj:
		if ident == "set" && len(args) == 2 {
			key := args[0].Inspect()
			l.Pairs[key] = args[1]
			return &Nil{}
		}
		if ident == "get" && len(args) == 1 {
			key := args[0].Inspect()
			if v, ok := l.Pairs[key]; ok {
				return v
			}
			return &Nil{}
		}
		// allow accessing fields via dot with no arg: map.key
		if len(argNodes) == 0 {
			if v, ok := l.Pairs[ident]; ok {
				return v
			}
			return &Nil{}
		}
	case *Str:
		if ident == "length" {
			return &Number{Value: float64(len(l.Value))}
		}
		if ident == "substring" && len(args) >= 2 {
			if a0, ok := args[0].(*Number); ok {
				if a1, ok2 := args[1].(*Number); ok2 {
					s := int(a0.Value)
					e := int(a1.Value)
					if s < 0 {
						s = 0
					}
					if e > len(l.Value) {
						e = len(l.Value)
					}
					return &Str{Value: l.Value[s:e]}
				}
			}
		}
	}
	return &Nil{}
}

// ---------- Utilities & Main ----------

func main() {
	if len(os.Args) > 1 {
		b, err := ioutil.ReadFile(os.Args[1])
		if err != nil {
			fmt.Println("error reading file:", err)
			return
		}
		src := string(b)
		runSource(src)
		return
	}
	// REPL
	fmt.Println("groovygo REPL — a tiny Groovy-like language in Go")
	env := NewEnvironment()
	registerBuiltins(env)
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print(">>> ")
		if !scanner.Scan() {
			break
		}
		line := scanner.Text()
		runLine(line, env)
	}
}

func runSource(src string) {
	l := NewLexer(src)
	p := NewParser(l)
	prog := p.ParseProgram()
	env := NewEnvironment()
	registerBuiltins(env)
	Eval(prog, env)
}

func runLine(line string, env *Environment) {
	l := NewLexer(line)
	p := NewParser(l)
	prog := p.ParseProgram()
	Eval(prog, env)
}
