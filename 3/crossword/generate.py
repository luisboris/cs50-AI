import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())


    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for v, words in self.domains.items():
            # delete words of different length
            for word in words.copy():
                if len(word) != v.length:
                    words.discard(word)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revision = False
        if self.crossword.overlaps[x, y] is not None:
            x_index, y_index = self.crossword.overlaps[x, y]
            for word1 in self.domains[x].copy():
                # search for a match in overlap, discard words without match
                match = False
                for word2 in self.domains[y].copy():
                    if word1[x_index] == word2[y_index]:
                        match = True
                        break
                if not match:
                    self.domains[x].discard(word1)
                    revision = True
                    
        return revision


    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        # begin with initial list of all arcs
        if not arcs:
            arcs = [
                (v1, v2) for (v1, v2), overlap in self.crossword.overlaps.items() 
                if overlap is not None
            ]
        
        # while there are still arcs to revise
        while len(arcs) != 0:
            # dequeue arc and revise
            x, y = arcs[0]
            arcs.pop(0)
            if self.revise(x, y):
                # no valid states
                if len(self.domains[x]) == 0:
                    return False
                # enqueue new arcs
                for z in self.crossword.neighbors(x):
                    if z is not y:
                        arcs.append((z, x))
        
        return True


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        if len(assignment) == len(self.domains):
            return True
        return False


    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        # list of words already assigned to variable
        assigned = []
        for v1, word1 in assignment.items():
            # all values are distinct and have the correct length
            if word1 in assigned or len(word1) != v1.length:
                return False
            # there are no conflicts between neighboring variables already in assignment
            for v2 in self.crossword.neighbors(v1):
                if v2 in assignment:
                    i, j = self.crossword.overlaps[v1, v2]
                    word2 = assignment[v2]
                    if word1[i] != word2[j]:
                        return False
            assigned.append(word1)

        return True


    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        values = []
        for word1 in self.domains[var]:
            # number of values discarded 
            count = 0
            # check for unnassigned variables among neighbors
            for v in self.domains:
                if v not in assignment and v in self.crossword.neighbors(var):
                    # count discards
                    var_index, v_index = self.crossword.overlaps[var, v]
                    for word2 in self.domains[v]:
                        if word1[var_index] != word2[v_index]:
                            count += 1
            values.append((word1, count))

        values.sort(key=lambda a: a[1])

        return [word for (word, count) in values]


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        # register chosen variable and corresponding number of values
        values = []
        for v in self.domains:
            if v not in assignment:
                values.append((v, len(self.domains[v])))
        values.sort(key=lambda a: a[1])

        # check for ties
        unnassigned = values[0][0]
        for (v, count) in values:
            if v == unnassigned:
                continue
            if values[0][1] == count:
                # find variable with most neighbors
                if self.crossword.neighbors(v) > self.crossword.neighbors(unnassigned):
                    unnassigned = v

        return unnassigned


    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        # assignment complete
        if self.assignment_complete(assignment):
            return assignment

        # get unnassigned variable
        v = self.select_unassigned_variable(assignment)

        # get consistent value
        for word in self.order_domain_values(v, assignment):
            assignment[v] = word
            # consistent
            if self.consistent(assignment):
                result = self.backtrack(assignment)
                if result is not None:
                    return result
                # not consistent
                assignment.pop(v)
        
        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
