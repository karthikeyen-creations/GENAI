import re
import os
from typing import Dict, List, Optional
from datetime import datetime
import glob

class DB2ToPostgresConverter:
    def __init__(self):
        # Previous mappings remain the same
        self.type_mappings = {
            'VARCHAR': 'VARCHAR',
            'CHAR': 'CHAR',
            'INTEGER': 'INTEGER',
            'SMALLINT': 'SMALLINT',
            'BIGINT': 'BIGINT',
            'DECIMAL': 'DECIMAL',
            'NUMERIC': 'NUMERIC',
            'TIMESTAMP': 'TIMESTAMP',
            'DATE': 'DATE',
            'CLOB': 'TEXT',
            'BLOB': 'BYTEA',
            'REAL': 'REAL',
            'DOUBLE': 'DOUBLE PRECISION',
            'BOOLEAN': 'BOOLEAN'
        }

        self.keyword_mappings = {
            'WITH RETURN': 'RETURNS SETOF',
            'LANGUAGE SQL': 'LANGUAGE plpgsql',
            'BEGIN': 'BEGIN',
            'END': 'END;',
            'DECLARE': 'DECLARE',
            'SET': 'SET',
        }

    def clean_sql(self, sql: str) -> str:
        """Clean and normalize SQL text."""
        # Replace multiple spaces with single space
        sql = re.sub(r'\s+', ' ', sql)
        
        # Remove spaces around parentheses and commas
        sql = re.sub(r'\s*\(\s*', '(', sql)
        sql = re.sub(r'\s*\)\s*', ')', sql)
        sql = re.sub(r'\s*,\s*', ', ', sql)
        
        # Normalize line endings
        sql = re.sub(r'[\r\n]+', '\n', sql)
        
        # Remove empty lines
        sql = '\n'.join(line for line in sql.splitlines() if line.strip())
        
        # Add proper spacing around keywords
        keywords = ['CREATE', 'PROCEDURE', 'FUNCTION', 'BEGIN', 'END', 'DECLARE', 'SET']
        for keyword in keywords:
            sql = re.sub(f'(?i)\\b{keyword}\\b', f' {keyword} ', sql)
            sql = re.sub(r'\s+', ' ', sql)
        
        return sql.strip()

    def parse_parameters(self, param_string: str) -> List[Dict[str, str]]:
        """Parse DB2 stored procedure parameters."""
        # Clean parameter string
        param_string = self.clean_sql(param_string)
        params = []
        param_list = param_string.strip().split(',')
        
        for param in param_list:
            param = param.strip()
            if not param:
                continue
                
            # Parse parameter components
            parts = param.split()
            param_info = {
                'name': parts[0],
                'direction': 'IN',
                'type': '',
                'size': ''
            }
            
            # Check for direction (IN, OUT, INOUT)
            if 'IN OUT' in param.upper() or 'INOUT' in param.upper():
                param_info['direction'] = 'INOUT'
            elif 'OUT' in param.upper():
                param_info['direction'] = 'OUT'
            elif 'IN' in param.upper():
                param_info['direction'] = 'IN'
                
            # Extract type information
            type_match = re.search(r'(\w+)(?:\((\d+(?:,\d+)?)\))?', param)
            if type_match:
                param_info['type'] = type_match.group(1).upper()
                if type_match.group(2):
                    param_info['size'] = type_match.group(2)
                    
            params.append(param_info)
            
        return params

    def convert_body(self, body: str) -> str:
        """Convert DB2 procedure body to PostgreSQL function body."""
        # Clean the body first
        body = self.clean_sql(body)
        
        # Replace common DB2 syntax patterns
        replacements = {
            r'WITH UR': '',
            r'SELECT\s+FROM': 'SELECT * FROM',
            r'VALUES\s+INTO': 'INTO',
            r'CURRENT DATE': 'CURRENT_DATE',
            r'CURRENT TIME': 'CURRENT_TIME',
            r'CURRENT TIMESTAMP': 'CURRENT_TIMESTAMP',
            r'CURRENT SERVER': 'CURRENT_DATABASE()',
            r'SUBSTR': 'SUBSTRING',
        }
        
        for pattern, replacement in replacements.items():
            body = re.sub(pattern, replacement, body, flags=re.IGNORECASE)
        
        # Add proper indentation
        lines = body.split('\n')
        indent_level = 1  # Start with one level after BEGIN
        indented_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Decrease indent for END statements
            if re.match(r'^\s*(END|END IF|END LOOP|END WHILE)', line, re.IGNORECASE):
                indent_level -= 1
            
            # Add indentation
            indented_lines.append('    ' * max(0, indent_level) + line)
            
            # Increase indent after BEGIN, IF, LOOP, WHILE statements
            if re.match(r'.*\s(BEGIN|IF|LOOP|WHILE)\s.*', line, re.IGNORECASE):
                indent_level += 1
            
        return '\n'.join(indented_lines)

    def convert_procedure(self, db2_procedure: str) -> str:
        """Convert DB2 stored procedure to PostgreSQL function."""
        # Clean input SQL
        db2_procedure = self.clean_sql(db2_procedure)
        
        # Extract procedure name
        name_match = re.search(r'CREATE\s+PROCEDURE\s+(\w+)', db2_procedure, re.IGNORECASE)
        if not name_match:
            raise ValueError("Could not find procedure name")
        
        proc_name = name_match.group(1)
        
        # Extract parameters
        param_match = re.search(r'\((.*?)\)', db2_procedure, re.DOTALL)
        if not param_match:
            params = []
        else:
            params = self.parse_parameters(param_match.group(1))
            
        # Extract procedure body
        body_match = re.search(r'BEGIN(.*?)END', db2_procedure, re.DOTALL)
        if not body_match:
            raise ValueError("Could not find procedure body")
            
        body = body_match.group(1).strip()
        converted_body = self.convert_body(body)
        
        # Build PostgreSQL function with proper formatting
        postgres_function = f"""CREATE OR REPLACE FUNCTION {proc_name}(
    {self.format_parameters(params)}
)
RETURNS {self.determine_return_type(db2_procedure)}
LANGUAGE plpgsql
AS $$
DECLARE
    -- Variable declarations
BEGIN
{converted_body}
END;
$$;"""
        
        return postgres_function

    def format_parameters(self, params: List[Dict[str, str]]) -> str:
        """Format parameters for PostgreSQL function."""
        formatted_params = []
        
        for param in params:
            pg_type = self.type_mappings.get(param['type'], param['type'])
            if param['size']:
                pg_type = f"{pg_type}({param['size']})"
                
            formatted_param = f"{param['name']} {param['direction']} {pg_type}"
            formatted_params.append(formatted_param)
            
        return ",\n    ".join(formatted_params)

    def determine_return_type(self, procedure: str) -> str:
        """Determine PostgreSQL function return type."""
        if 'WITH RETURN' in procedure.upper():
            return "TABLE (...)"
        elif 'RETURNS' in procedure.upper():
            return_match = re.search(r'RETURNS\s+(\w+)', procedure, re.IGNORECASE)
            if return_match:
                return self.type_mappings.get(return_match.group(1).upper(), 'void')
        return 'void'

class FileHandler:
    def __init__(self):
        self.input_folder = "DB2SPs"
        self.output_base_folder = "pgSQL"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = os.path.join(self.output_base_folder, self.timestamp)

    def setup_folders(self):
        """Create necessary folders if they don't exist."""
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)

    def get_input_files(self) -> List[str]:
        """Get list of all .sql files in the input folder."""
        return glob.glob(os.path.join(self.input_folder, "*.sql"))

    def read_input_file(self, file_path: str) -> str:
        """Read content from input SQL file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove any BOM characters
                return content.strip('\ufeff')
        except Exception as e:
            raise IOError(f"Error reading file {file_path}: {str(e)}")

    def write_output_file(self, content: str, original_filename: str):
        """Write converted content to output SQL file."""
        output_path = os.path.join(self.output_folder, os.path.basename(original_filename))
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Successfully written to {output_path}")
        except Exception as e:
            raise IOError(f"Error writing to file {output_path}: {str(e)}")

def main():
    converter = DB2ToPostgresConverter()
    file_handler = FileHandler()
    
    try:
        # Setup folder structure
        file_handler.setup_folders()
        
        # Get list of input files
        input_files = file_handler.get_input_files()
        
        if not input_files:
            print(f"No SQL files found in {file_handler.input_folder} folder")
            return
        
        print(f"Found {len(input_files)} SQL files to process")
        print(f"Output folder: {file_handler.output_folder}")
        
        # Process each file
        for input_file in input_files:
            try:
                print(f"\nProcessing {input_file}...")
                
                # Read and clean input file
                db2_procedure = file_handler.read_input_file(input_file)
                
                # Convert procedure
                postgres_function = converter.convert_procedure(db2_procedure)
                
                # Write output file
                file_handler.write_output_file(postgres_function, input_file)
                
            except Exception as e:
                print(f"Error processing {input_file}: {str(e)}")
                continue
        
        print("\nConversion process completed!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()