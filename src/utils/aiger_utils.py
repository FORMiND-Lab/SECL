try:
    from .aig_schema import GATE_TO_INDEX
except ImportError:
    from aig_schema import GATE_TO_INDEX

def _canonical_gate_mapping(gate_to_index=None):
    if gate_to_index is None:
        return GATE_TO_INDEX
    normalized = {str(key): int(value) for key, value in gate_to_index.items()}
    if normalized != GATE_TO_INDEX:
        raise ValueError(
            f"SECL graph schema requires {GATE_TO_INDEX}, got {normalized}"
        )
    return GATE_TO_INDEX

def aig_to_xdata(aig_filename, gate_to_index=None):
    gate_to_index = _canonical_gate_mapping(gate_to_index)

    def decode_delta(f):
        result = 0
        shift = 0
        while True:
            byte = f.read(1)
            if len(byte) != 1:
                raise ValueError("Unexpected end of file")
            byte = byte[0]
            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                break
            shift += 7
        return result

    with open(aig_filename, 'rb') as f:

        header = f.readline().decode('ascii').strip()
        parts = header.split()

        if parts[0] != 'aig':
            raise ValueError(f"Invalid AIG file format: {header}")


        n_variables = int(parts[1])
        n_inputs = int(parts[2])
        n_latches = int(parts[3])
        n_outputs = int(parts[4])
        n_and = int(parts[5])

        if n_latches != 0:
            raise ValueError('The AIG has latches, which are not supported')
        if n_variables != n_inputs + n_and:
            raise ValueError(
                "SECL requires compact combinational binary AIGER numbering: "
                f"M={n_variables}, I+A={n_inputs + n_and}"
            )


        x_data = [[0, gate_to_index['CONST']]]
        edge_index = []


        for i in range(n_inputs):
            x_data.append([len(x_data), gate_to_index['PI']])


        for i in range(n_and):
            x_data.append([len(x_data), gate_to_index['AND']])


        outputs = []
        for i in range(n_outputs):
            line = f.readline().decode('ascii').strip()
            literal = int(line)
            if not 0 <= literal <= 2 * n_variables + 1:
                raise ValueError(f"PO literal {literal} is outside AIGER range")
            outputs.append(literal)


        not_dict = {}


        and_gates = []
        for i in range(n_and):
            output_literal = (n_inputs + 1 + i) * 2


            delta1 = decode_delta(f)
            input1_literal = output_literal - delta1

            delta2 = decode_delta(f)
            input2_literal = input1_literal - delta2

            for input_literal in (input1_literal, input2_literal):
                if not 0 <= input_literal < output_literal:
                    raise ValueError(
                        f"AND literal {output_literal} has invalid fanin {input_literal}"
                    )

            and_gates.append((output_literal, input1_literal, input2_literal))


        for output_literal, input1_literal, input2_literal in and_gates:
            output_idx = output_literal // 2


            input1_idx = input1_literal // 2
            sign1 = input1_literal % 2
            if sign1 == 1:
                if input1_idx in not_dict:
                    not1_idx = not_dict[input1_idx]
                else:
                    x_data.append([len(x_data), gate_to_index['NOT']])
                    not1_idx = len(x_data) - 1
                    not_dict[input1_idx] = not1_idx
                    edge_index.append([input1_idx, not1_idx])
                input1_final = not1_idx
            else:
                input1_final = input1_idx


            input2_idx = input2_literal // 2
            sign2 = input2_literal % 2
            if sign2 == 1:
                if input2_idx in not_dict:
                    not2_idx = not_dict[input2_idx]
                else:
                    x_data.append([len(x_data), gate_to_index['NOT']])
                    not2_idx = len(x_data) - 1
                    not_dict[input2_idx] = not2_idx
                    edge_index.append([input2_idx, not2_idx])
                input2_final = not2_idx
            else:
                input2_final = input2_idx


            edges_to_add = [(input1_final, output_idx), (input2_final, output_idx)]
            edges_to_add.sort()
            for edge in edges_to_add:
                edge_index.append(list(edge))


        for output_literal in outputs:
            if output_literal % 2 == 1:
                output_idx = output_literal // 2
                if output_idx not in not_dict:
                    x_data.append([len(x_data), gate_to_index['NOT']])
                    not_idx = len(x_data) - 1
                    not_dict[output_idx] = not_idx
                    edge_index.append([output_idx, not_idx])

        return x_data, edge_index

def myaig_to_xdata(aig_filename, gate_to_index=None):
    x_data, edge_index = aig_to_xdata(aig_filename, gate_to_index)
    gate_to_index = _canonical_gate_mapping(gate_to_index)
    base_types = {
        gate_to_index['CONST'], gate_to_index['PI'], gate_to_index['AND']
    }
    num_base_nodes = sum(1 for x in x_data if x[1] in base_types)
    return x_data, edge_index, num_base_nodes
