use colored::Colorize;
use regex::Regex;
use std::collections::{self, VecDeque};
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};

use lazy_static::lazy_static;

lazy_static! {
    static ref OP_MAP: HashMap<&'static str, i32> =
        HashMap::from([("^", 4), ("*", 3), ("/", 3), ("+", 2), ("-", 2),]);
    static ref FUNCTION_SET: HashSet<&'static str> =
        HashSet::from(["sin", "cos", "tan", "sqrt", "log", "ln", "abs"]);
}

fn calculate(input: &mut String) -> Result<f32, String> {
    let mut tokens = Vec::new();
    let variables: HashMap<String, f32> = HashMap::new();
    match tokenize(input, &mut tokens, &variables) {
        Ok(_) => (),
        Err(err) => {
            return Err(err);
        }
    };

    let ops = match shunting_yardify(&tokens) {
        Ok(ops) => ops,
        Err(err) => {
            return Err(format!("Syntax error: {}", err));
        }
    };

    let out = match eval_reverse_polish(&ops) {
        Ok(out) => out,
        Err(err) => {
            return Err(err);
        }
    };
    return Ok(out);
}

fn calculate_from_tokens(tokens: &Vec<String>) -> Result<f32, String> {
    let ops = match shunting_yardify(tokens) {
        Ok(ops) => ops,
        Err(err) => {
            return Err(format!("Syntax error: {}", err));
        }
    };

    let out = match eval_reverse_polish(&ops) {
        Ok(out) => out,
        Err(err) => {
            return Err(err);
        }
    };
    return Ok(out);
}

fn main() {
    let mut buffer = String::new();
    let mut step = 0;
    let mut variables: HashMap<String, f32> = HashMap::new();

    loop {
        print!("calc({})> ", step);
        step += 1;
        io::stdout().flush().unwrap();
        io::stdin().read_line(&mut buffer).unwrap();

        buffer = buffer.to_lowercase();

        let mut tokens = Vec::new();
        match tokenize(&buffer, &mut tokens, &variables) {
            Ok(_) => (),
            Err(err) => {
                buffer.clear();
                println!("{}", err.red().bold());
                continue;
            }
        };

        let mut variable_name: Option<String> = None;
        if tokens[0].contains('=') {
            let beginning = tokens.remove(0);
            variable_name = Some(
                beginning
                    .split('=')
                    .next()
                    .unwrap()
                    .to_string()
                    .trim()
                    .to_string(),
            );
        }

        let out = match calculate_from_tokens(&tokens) {
            Ok(out) => out,
            Err(err) => {
                buffer.clear();
                println!("{}", err.red().bold());
                continue;
            }
        };

        buffer.clear();

        match variable_name {
            Some(name) => {
                variables.insert(name.clone(), out.clone());
                println!(
                    "{} = {}",
                    name.bold().yellow(),
                    out.to_string().bold().yellow()
                );
            }
            None => {
                println!("{}", out.to_string().bold().yellow());
            }
        }
    }
}

fn tokenize(
    input: &str,
    output: &mut Vec<String>,
    variables: &HashMap<String, f32>,
) -> Result<(), String> {
    let num_re = Regex::new(r"-?\d+(\.\d+)?").unwrap();
    let paren_re = Regex::new(r"[\(\)]").unwrap();
    let ops_re = Regex::new(r"[\*\+\-\/\^]").unwrap();
    let funcs_re = Regex::new(r"\b(?:sin|cos|tan|sqrt|log|ln|abs)\b").unwrap();
    let symbols_re = Regex::new(r"\b(?:pi|e)\b").unwrap();
    let variables_re = Regex::new(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=").unwrap();
    let var_name_re = Regex::new(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b").unwrap();

    let symbols_map = HashMap::from([("pi", std::f32::consts::PI), ("e", std::f32::consts::E)]);

    let mut i: usize = 0;
    while i < input.len() {
        if input.chars().nth(i).unwrap().is_whitespace() {
            i += 1;
            continue;
        }

        if let Some(vars) = variables_re.find_at(input, i) {
            if vars.start() == i {
                output.push(input[vars.start()..vars.end()].to_string());
                i = vars.end();
                continue;
            }
        }

        if let Some(funcs) = funcs_re.find_at(input, i) {
            if funcs.start() == i {
                output.push(input[funcs.start()..funcs.end()].to_string());
                i = funcs.end();
                continue;
            }
        }

        if let Some(nums) = num_re.find_at(input, i) {
            if nums.start() == i {
                output.push(input[nums.start()..nums.end()].to_string());
                i = nums.end();
                continue;
            }
        }

        if let Some(parens) = paren_re.find_at(input, i) {
            if parens.start() == i {
                output.push(input[parens.start()..parens.end()].to_string());
                i = parens.end();
                continue;
            }
        }

        if let Some(ops) = ops_re.find_at(input, i) {
            if ops.start() == i {
                output.push(input[ops.start()..ops.end()].to_string());
                i = ops.end();
                continue;
            }
        }

        if let Some(ops) = symbols_re.find_at(input, i) {
            if ops.start() == i {
                let symbol_value = symbols_map.get(&input[ops.start()..ops.end()]).unwrap();
                output.push(symbol_value.to_string());
                i = ops.end();
                continue;
            }
        }

        if let Some(var_name) = var_name_re.find_at(input, i) {
            if var_name.start() == i {
                let value = variables
                    .get(&input[var_name.start()..var_name.end()])
                    .ok_or_else(|| {
                        format!(
                            "Variable \"{}\" not found",
                            input[var_name.start()..var_name.end()].to_string()
                        )
                    })?;
                output.push(value.to_string());
                i = var_name.end();
                continue;
            }
        }

        return Err(format!(
            "Invalid token \"{}\" at index {}",
            input.chars().nth(i).unwrap(),
            i
        ));
    }
    return Ok(());
}

fn eval_reverse_polish(input: &Vec<String>) -> Result<f32, String> {
    let mut stack: VecDeque<f32> = VecDeque::new();

    for token in input {
        match token.as_str() {
            "*" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                let l = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(l * r);
            }
            "-" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                let l = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(l - r);
            }
            "^" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                let l = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(l.powf(r));
            }
            "+" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                let l = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(l + r);
            }
            "/" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                let l = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(l / r);
            }
            "sqrt" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(r.sqrt());
            }
            "log" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(r.log(10.0));
            }
            "ln" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(r.ln());
            }
            "abs" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(r.abs());
            }
            "sin" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(r.sin());
            }
            "cos" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(r.cos());
            }
            "tan" => {
                let r = stack.pop_back().ok_or("Syntax error: invalid input")?;
                stack.push_back(r.tan());
            }
            _ => {
                stack.push_back(
                    token
                        .parse::<f32>()
                        .map_err(|_| format!("Syntax error: unrecognized token \"{}\"", token))?,
                );
            }
        }
    }

    if stack.len() == 1 {
        Ok(stack.pop_back().unwrap())
    } else {
        Err("Invalid input".to_string())
    }
}

#[derive(Debug)]
enum ShuntingErrors {
    MismatchedParenthesis,
    InvalidInput(String),
}

impl std::fmt::Display for ShuntingErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShuntingErrors::MismatchedParenthesis => write!(f, "Mismatched parenthesis"),
            ShuntingErrors::InvalidInput(token) => write!(f, "Invalid token \"{}\"", token),
        }
    }
}

fn shunting_yardify(input: &Vec<String>) -> Result<Vec<String>, ShuntingErrors> {
    let mut output: Vec<String> = Vec::new();
    let mut op_stack: VecDeque<String> = collections::VecDeque::new();

    for token in input {
        let op_type = get_optype(token);
        match op_type {
            OpType::Number => output.push(token.to_string()),
            OpType::Operator => handle_operator(token, &mut op_stack, &mut output)?,
            OpType::Function => op_stack.push_back(token.to_string()),
            OpType::LeftParen => op_stack.push_back(token.to_string()),
            OpType::RightParen => handle_right_paren(&mut op_stack, &mut output)?,
            OpType::Variable => println!("Variable: {}", token),
        }
    }

    while !op_stack.is_empty() {
        let back = op_stack.pop_back().unwrap();

        if back == "(" {
            return Err(ShuntingErrors::MismatchedParenthesis);
        }

        output.push(back);
    }

    return Ok(output);
}

fn handle_operator(
    token: &String,
    stack: &mut VecDeque<String>,
    output: &mut Vec<String>,
) -> Result<(), ShuntingErrors> {
    let o1_prec = OP_MAP
        .get(token.as_str())
        .ok_or_else(|| ShuntingErrors::InvalidInput(token.to_string()))?;

    while let Some(back) = stack.back() {
        if back == "(" {
            break;
        }

        if get_optype(back) == OpType::Function {
            break;
        }

        let o2_prec = OP_MAP
            .get(back.as_str())
            .ok_or_else(|| ShuntingErrors::InvalidInput(back.to_string()))?;

        if o2_prec > o1_prec || (o1_prec == o2_prec && token != "^") {
            output.push(stack.pop_back().unwrap());
        } else {
            break;
        }
    }
    stack.push_back(token.to_string());
    Ok(())
}

fn handle_right_paren(
    stack: &mut VecDeque<String>,
    output: &mut Vec<String>,
) -> Result<(), ShuntingErrors> {
    while let Some(back) = stack.pop_back() {
        if back == "(" {
            return Ok(());
        }
        output.push(back);
    }
    Err(ShuntingErrors::MismatchedParenthesis)
}

#[derive(PartialEq, Eq)]
enum OpType {
    Number,
    Operator,
    Function,
    LeftParen,
    RightParen,
    Variable,
}

fn get_optype(token: &String) -> OpType {
    if token.parse::<f32>().is_ok() {
        return OpType::Number;
    } else if token == "(" {
        return OpType::LeftParen;
    } else if token == ")" {
        return OpType::RightParen;
    } else if OP_MAP.contains_key(&token.as_str()) {
        return OpType::Operator;
    } else if FUNCTION_SET.contains(&token.as_str()) {
        return OpType::Function;
    }
    return OpType::Variable;
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn addition() {
        let mut input = String::from("1 + 2");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 3.0);
    }

    #[test]
    fn subtraction() {
        let mut input = String::from("1 - 2");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, -1.0);
    }

    #[test]
    fn multiplication() {
        let mut input = String::from("1 * 2");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 2.0);
    }

    #[test]
    fn division() {
        let mut input = String::from("1 / 2");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 0.5);
    }

    #[test]
    fn mult_and_subtract() {
        let mut input = String::from("2 * 3 - 1");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 5.0);
    }
    #[test]
    fn parens_division() {
        let mut input = String::from("(8 + 2) / 5");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 2.0);
    }
    #[test]
    fn exponent() {
        let mut input = String::from("2^3");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 8.0);
    }
    #[test]
    fn complex_ops() {
        let mut input = String::from("(3 + 5) * 2 ^ (4 - 2) / 8");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 4.0);
    }
    #[test]
    fn nested_parens() {
        let mut input = String::from("((2 + 3) * (5 - 2)) / (4 ^ 2)");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 0.9375);
    }
    #[test]
    fn mixed_negative() {
        let mut input = String::from("5 - 3 * 2 + 1");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 0.0);
    }

    #[test]
    fn pi() {
        let mut input = String::from("pi");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, std::f32::consts::PI);
    }

    #[test]
    fn e() {
        let mut input = String::from("e");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, std::f32::consts::E);
    }

    #[test]
    fn sin() {
        let mut input = String::from("sin(pi)");
        let out = calculate(&mut input).unwrap();
        assert_relative_eq!(out, 0.0, epsilon = f32::EPSILON);
    }

    #[test]
    fn cos() {
        let mut input = String::from("cos(pi)");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, -1.0);
    }

    #[test]
    fn sqrt() {
        let mut input = String::from("sqrt(4)");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 2.0);
    }

    #[test]
    fn log() {
        let mut input = String::from("log(10)");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 1.0);
    }

    #[test]
    fn ln() {
        let mut input = String::from("ln(10)");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 10.0_f32.ln());
    }

    #[test]
    fn abs() {
        let mut input = String::from("abs(-10)");
        let out = calculate(&mut input).unwrap();
        assert_eq!(out, 10.0);
    }

    #[test]
    fn function_multiply() {
        let mut input = String::from("sin pi * 4");
        let out = calculate(&mut input).unwrap();
        assert_relative_eq!(out, 0.0, epsilon = f32::EPSILON * 4.0);
    }

    #[test]
    fn variable_assign() {
        let input = String::from("a = pi * 2\n");
        let mut tokens = Vec::new();
        let variables: HashMap<String, f32> = HashMap::new();
        match tokenize(&input, &mut tokens, &variables) {
            Ok(_) => (),
            Err(err) => {
                println!("{}", err);
            }
        };
        let out = calculate_from_tokens(&tokens).unwrap();
        assert_relative_eq!(out, 6.283185307179586, epsilon = f32::EPSILON);
    }
}
