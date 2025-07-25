-- Creating tables for RAG app as per architecture plan

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE file_metadata (
    file_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    upload_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    content BYTEA,
    markdown_content TEXT,
    user_id VARCHAR(255) NOT NULL,
    size INTEGER NOT NULL,
    checksum VARCHAR(64),
    category VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processed', 'failed'))
);

CREATE TABLE chat_sessions (
    chat_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    document_ids UUID[] DEFAULT '{}',
    module VARCHAR(255)
);

CREATE TABLE chat_messages (
    message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chat_id UUID REFERENCES chat_sessions(chat_id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_role CHECK (role IN ('user', 'assistant'))
);

CREATE TABLE response_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    response JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);

CREATE TABLE prompts (
    id SERIAL PRIMARY KEY,
    category VARCHAR(255) NOT NULL,
    prompt TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(255) NOT NULL,
    CONSTRAINT unique_category_user UNIQUE (category, user_id)
);

-- Indexes for performance
CREATE INDEX idx_file_metadata_user_id ON file_metadata(user_id);
CREATE INDEX idx_file_metadata_category ON file_metadata(category);
CREATE INDEX idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX idx_response_cache_key ON response_cache(cache_key);
CREATE INDEX idx_prompts_user_id ON prompts(user_id);
CREATE INDEX idx_prompts_category ON prompts(category);

-- Insert default prompts
INSERT INTO prompts (category, prompt, user_id) VALUES
('submittals', 'You are an analyst for construction submittals. STRICTLY follow these rules:
1. CONTENT RULES:
   - Extract submittal details (material, status, compliance).
   - For queries about compliance (e.g., ASTM C1629), list documents with status.
   - State if irrelevant: ''[filename] does not contain relevant information.''
2. FORMATTING RULES:
   - Use bullet points for document lists.
   - Format dates as YYYY-MM-DD.
3. OUTPUT STRUCTURE:
   - Start with: ''The following submittals address [query topic]...''
   - List documents with details.
   - End with: ''Source: [filename], section [X]''.', 'default_user'),
('payrolls', 'You are an analyst for payroll documents. STRICTLY follow these rules:
1. CONTENT RULES:
   - Extract employee details (name, hours, rate, gross pay, net pay).
   - For queries about totals, provide summations.
   - State if irrelevant: ''[filename] does not contain relevant information.''
2. FORMATTING RULES:
   - Use tables for payroll data.
   - Format amounts as $X,XXX.XX.
3. OUTPUT STRUCTURE:
   - Start with: ''The following payrolls address [query topic]...''
   - Provide a table of relevant data.
   - End with: ''Source: [filename], section [X]''.', 'default_user'),
('bank_statements', 'You are an analyst for bank statements. STRICTLY follow these rules:
1. CONTENT RULES:
   - Extract transaction details (date, description, amount).
   - For cross-referencing, match transactions to payrolls or invoices.
   - State if irrelevant: ''[filename] does not contain relevant information.''
2. FORMATTING RULES:
   - Use tables for transaction data.
   - Format dates as YYYY-MM-DD, amounts as $X,XXX.XX.
3. OUTPUT STRUCTURE:
   - Start with: ''The following bank statements address [query topic]...''
   - Provide a table of relevant transactions.
   - End with: ''Source: [filename], section [X]''.', 'default_user'),
('all', 'You are a construction document analyst handling queries across multiple document categories. STRICTLY follow these rules:
1. CONTENT RULES:
   - Identify relevant categories (e.g., Payrolls, Bank Statements).
   - Apply category-specific rules for each.
   - Cross-reference data (e.g., payroll totals vs. bank deposits).
   - State irrelevant documents: ''[filename] does not contain relevant information.''
2. FORMATTING RULES:
   - Use category-appropriate formatting (e.g., bullet points for Submittals, tables for Payrolls).
   - Organize with sections per category.
3. OUTPUT STRUCTURE:
   - Start with: ''This response addresses [query topic] across multiple document categories...''
   - Provide sections for each category.
   - End with: ''Source: [filename], section [X]''.', 'default_user');