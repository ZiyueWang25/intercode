"""Helps agent retrieve relevant code snippets"""
from collections.abc import Sequence, Mapping
from typing import Union

from docker.models.containers import Container

class Line:
	def __init__(self, graph, path: str, line_number: str, content: str):
		self.graph = graph
		self.path = path
		self.line_number = line_number
		self.content = content
	
	def __repr__(self) -> str:
		return f"{self.path}:{self.line_number} {self.content}"

	def __lt__(self, other: 'Line'):
		if self.path < other.path:
			return True
		return self.line_number < other.line_number
	
	def get_around(self, num:int=3):
		"""Uses the graph to get the surrounding context."""
		raise NotImplementedError

class Symbol:
	def __init__(self, full_name: str):
		self.full_name = full_name
		self._convert()
		self.line_number = None

	def _convert(self):
		self.path = self.full_name.split(":")[0]
		self.symbol_name = self.full_name[len(self.path)+1:]
	
	def __repr__(self) -> str:
		return self.full_name
	
	def __lt__(self, other: "Symbol") -> bool:
		if self.path < other.path:
			return True
		return self.line_number < other.line_number

	def __eq__(self, other: "Symbol") -> bool:
		return str(self) == str(other)


class CodeGraph:
	def __init__(self, container:Container, repo_name:str):
		self.container = container
		self.workdir = f"/{repo_name}/"
		self.graph = None
		self.symbols_by_filepath: Mapping[str, Sequence[Symbol]] = {}
		self.symbol_by_fullname: Mapping[str, Symbol] = {}

	def update(self):
		self._built_graph()
	
	def _built_graph(self):
		"""Builds the code graph of the current repository"""
		return NotImplementedError
	
	def get_code_tree(self, filepath:str):
		"""Returns the tree format of class and functions in the file.
		
		Each line starts with the line number in the current file. Then it 
		For example:
		>> code_graph.get_code_tree("/intercode/intercode/envs/swe/swe_env.py")
		'''Code Tree:
		16 copy_to_container
		26 SWEEnv
		30   __init__
		37   reset_container
		69   step
		126   apply_patch
		154   extract_command
		160   extract_patch
		166   get_reward
		172   close'''
		"""		
		if filepath not in self.symbols_by_filepath:
			raise ValueError(f"filepath {filepath!r} doesn't exist in {self.workdir}")
		lines = ["Code Tree:"]
		symbols = sorted(symbols)
		for symbol in symbols:
			line = f"{symbol.line_number} "
			symbol_name = symbol.symbol_name
			symbol_name_abbr = symbol_name.split("::")[-1]
			num_tab = len(symbol_name.split("::"))
			line += num_tab * "\t"
			line += symbol_name_abbr
			lines.append(line)
		return "\n".join(lines)

	def get_all_reference(self, symbol: str, num_around: int) -> str:
		if isinstance(symbol, str):
			symbol = Symbol(symbol)
		all_references = self.graph.get_all_reference(symbol)
		lines = self._parse_reference(all_references)
		return self._show_lines(lines, num_around)
	
	def _parse_reference(self, all_references) -> Sequence[Line]:
		raise NotImplementedError
	
	def _show_lines(self, lines:Sequence[Line], num_around) -> str:
		lines = sorted(lines)
		result = []
		for i, line in enumerate(lines):
			result.append(f"Reference {i}")
			result.append(line.get_around(num_around))
		return "\n".join(result)
