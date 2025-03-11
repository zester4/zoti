
// Import dotenv to load environment variables from .env file
import * as dotenv from 'dotenv';
dotenv.config();

import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatGroq } from "@langchain/groq";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import * as readline from 'readline';
import * as fs from 'fs';
import * as path from 'path';
import { Tool } from "@langchain/core/tools";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
// Import AWS SDK for Polly
import { Polly } from "@aws-sdk/client-polly";
import { createWriteStream } from 'fs';
import { spawn } from 'child_process';
import { pipeline } from 'stream/promises';

// Log the environment variables (not the values) to confirm they're loaded
console.log('Environment variables loaded:');
console.log('- GOOGLE_API_KEY:', process.env.GOOGLE_API_KEY ? '✅ Found' : '❌ Missing');
console.log('- TAVILY_API_KEY:', process.env.TAVILY_API_KEY ? '✅ Found' : '❌ Missing');
console.log('- GROQ_API_KEY:', process.env.GROQ_API_KEY ? '✅ Found' : '❌ Missing');
console.log('- AWS_ACCESS_KEY_ID:', process.env.AWS_ACCESS_KEY_ID ? '✅ Found' : '❌ Missing');
console.log('- AWS_SECRET_ACCESS_KEY:', process.env.AWS_SECRET_ACCESS_KEY ? '✅ Found' : '❌ Missing');
console.log('- AWS_REGION:', process.env.AWS_REGION ? '✅ Found' : '❌ Missing');

// Validate required API keys before proceeding
if (!process.env.TAVILY_API_KEY) {
  console.error('❌ ERROR: TAVILY_API_KEY not found in environment variables.');
  console.error('Please add it to your .env file:');
  console.error('TAVILY_API_KEY=tvly-...');
  process.exit(1);
}

if (!process.env.GROQ_API_KEY) {
  console.error('❌ ERROR: GROQ_API_KEY not found in environment variables.');
  console.error('Please add it to your .env file:');
  console.error('GROQ_API_KEY=gsk-...');
  process.exit(1);
}

if (!process.env.AWS_ACCESS_KEY_ID || !process.env.AWS_SECRET_ACCESS_KEY || !process.env.AWS_REGION) {
  console.error('❌ ERROR: AWS credentials not found in environment variables.');
  console.error('Please add them to your .env file:');
  console.error('AWS_ACCESS_KEY_ID=...');
  console.error('AWS_SECRET_ACCESS_KEY=...');
  console.error('AWS_REGION=us-east-1 (or your preferred region)');
  console.error('Voice functionality will be disabled.');
}

// Create a readline interface for user input
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Document storage object to maintain state across the session
const documentStore = {
  currentDocument: null as any,
  documentPages: [] as any[],
  currentPageIndex: 0,
  documentName: '',
  totalPages: 0
};

// Voice configuration storage
const voiceConfig = {
  enabled: true,
  selectedVoice: 'Matthew', // Default voice
  availableVoices: {
    male: ['Matthew', 'Kevin', 'Stephen'],
    female: ['Joanna', 'Kimberly', 'Salli']
  }
};

// Initialize AWS Polly client if credentials are available
let pollyClient: Polly | null = null;
if (process.env.AWS_ACCESS_KEY_ID && process.env.AWS_SECRET_ACCESS_KEY && process.env.AWS_REGION) {
  pollyClient = new Polly({
    region: process.env.AWS_REGION,
    credentials: {
      accessKeyId: process.env.AWS_ACCESS_KEY_ID,
      secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
    }
  });
}

// Function to synthesize speech using Amazon Polly - Updated to hide debug messages
async function synthesizeSpeech(text: string, voiceId: string): Promise<void> {
  if (!pollyClient || !voiceConfig.enabled) {
    return; // Skip if Polly client is not initialized or voice is disabled
  }

  try {
    // Create output directory if it doesn't exist
    const outputDir = path.join(__dirname, 'audio');
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }

    // Prepare output file path
    const timestamp = Date.now();

    // Break text into chunks if needed (Amazon Polly has a 3000 character limit)
    const maxChunkSize = 3000;
    const textChunks = [];

    // Simple chunking by sentences
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    let currentChunk = '';

    for (const sentence of sentences) {
      if (currentChunk.length + sentence.length > maxChunkSize) {
        textChunks.push(currentChunk);
        currentChunk = sentence;
      } else {
        currentChunk += sentence;
      }
    }

    if (currentChunk) {
      textChunks.push(currentChunk);
    }

    // Process each chunk - no longer logging debug information
    for (let i = 0; i < textChunks.length; i++) {
      const chunk = textChunks[i];
      // Use plain text instead of SSML for simplicity
      const chunkFile = path.join(outputDir, `speech_${timestamp}_${i}.mp3`);
      
      // Configure the Polly synthesis request
      const params = {
        OutputFormat: "mp3",
        Text: chunk,
        TextType: "text",
        VoiceId: voiceId,
        Engine: "neural"
      };

      // Get audio stream from Polly
      const response = await pollyClient.synthesizeSpeech(params);
      
      if (response.AudioStream) {
        // Create write stream
        const writeStream = createWriteStream(chunkFile);
        
        // Use pipeline for proper stream handling
        await pipeline(
          response.AudioStream as any,
          writeStream
        );
        
        // Play the audio without logging
        await playAudio(chunkFile);
      }
    }
  } catch (error) {
    // Only log critical errors
    console.error('Error synthesizing speech - check AWS credentials and configuration');
  }
}

// Function to play audio files based on operating system - Updated to hide debug messages
async function playAudio(audioFile: string): Promise<void> {
  return new Promise((resolve, reject) => {
    let player;
    
    try {
      switch (process.platform) {
        case 'win32':
          // For Windows, use the simpler start command
          player = spawn('cmd.exe', ['/c', `start /wait "" "${audioFile}"`], {
            shell: true,
            stdio: 'ignore'
          });
          break;
        case 'darwin': // macOS
          player = spawn('afplay', [audioFile]);
          break;
        default: // Linux and others
          // Try mpg123 first, fallback to others if not available
          try {
            player = spawn('mpg123', ['-q', audioFile]);
          } catch (error) {
            try {
              player = spawn('mplayer', ['-really-quiet', audioFile]);
            } catch (error) {
              player = spawn('play', [audioFile]);
            }
          }
          break;
      }
      
      player.on('error', (err) => {
        // Only show error on first occurrence, not for every message
        if (!playAudio.hasOwnProperty('errorShown')) {
          console.error(`Error playing audio: ${err.message}`);
          console.error('Make sure you have the appropriate audio player installed for your OS:');
          console.error('- Windows: built-in audio player');
          console.error('- macOS: afplay (included with macOS)');
          console.error('- Linux: mpg123, mplayer, or sox (install with your package manager)');
          (playAudio as any).errorShown = true;
        }
        resolve(); // Resolve anyway to continue execution
      });
      
      player.on('close', (code) => {
        resolve();
      });
    } catch (error) {
      resolve(); // Continue execution even if playing fails
    }
  });
}


// Custom document loading tool
class DocumentLoadTool extends Tool {
    name = "document_loader";
    description = "Load a PDF or DOCX document for analysis and teaching. Input should be the file path.";
  
    async _call(filePath: string): Promise<string> {
      try {
        // Check if file exists
        if (!fs.existsSync(filePath)) {
          return `Error: File not found at path "${filePath}". Please provide a valid file path.`;
        }
  
        const fileExt = path.extname(filePath).toLowerCase();
        let loader;
        let docs;
  
        // Load document based on file extension
        if (fileExt === '.pdf') {
          loader = new PDFLoader(filePath);
          docs = await loader.load();
        } else if (fileExt === '.docx') {
          loader = new DocxLoader(filePath);
          docs = await loader.load();
        } else {
          return `Error: Unsupported file format "${fileExt}". Only PDF and DOCX files are supported.`;
        }
  
        // Store document in the document store
        documentStore.documentPages = docs;
        documentStore.currentPageIndex = 0;
        documentStore.totalPages = docs.length;
        documentStore.documentName = path.basename(filePath);
  
        return `Successfully loaded ${documentStore.documentName} with ${documentStore.totalPages} pages. You can now use the page_reader tool to explore the document page by page. Start with "read_current_page" to view the first page.`;
      } catch (error) {
        console.error('Error loading document:', error);
        return `Error loading document: ${error.message}`;
      }
    }
  }
  
  // Custom page reader tool
  class PageReaderTool extends Tool {
    name = "page_reader";
    description = "Read pages from the loaded document. Input should be one of: 'read_current_page', 'next_page', 'previous_page', 'go_to_page:{number}', 'document_summary', 'page_count'.";
  
    async _call(command: string): Promise<string> {
      // Check if a document is loaded
      if (!documentStore.documentPages || documentStore.documentPages.length === 0) {
        return "No document is currently loaded. Please use the document_loader tool first.";
      }
  
      const commandLower = command.toLowerCase();
  
      try {
        // Handle different commands
        if (commandLower === 'read_current_page') {
          const currentPage = documentStore.documentPages[documentStore.currentPageIndex];
          return `[Page ${documentStore.currentPageIndex + 1}/${documentStore.totalPages} of "${documentStore.documentName}"]\n\n${currentPage.pageContent}`;
        }
        else if (commandLower === 'next_page') {
          if (documentStore.currentPageIndex < documentStore.totalPages - 1) {
            documentStore.currentPageIndex++;
            const currentPage = documentStore.documentPages[documentStore.currentPageIndex];
            return `[Page ${documentStore.currentPageIndex + 1}/${documentStore.totalPages} of "${documentStore.documentName}"]\n\n${currentPage.pageContent}`;
          } else {
            return `You are already at the last page (${documentStore.totalPages}) of the document.`;
          }
        }
        else if (commandLower === 'previous_page') {
          if (documentStore.currentPageIndex > 0) {
            documentStore.currentPageIndex--;
            const currentPage = documentStore.documentPages[documentStore.currentPageIndex];
            return `[Page ${documentStore.currentPageIndex + 1}/${documentStore.totalPages} of "${documentStore.documentName}"]\n\n${currentPage.pageContent}`;
          } else {
            return `You are already at the first page of the document.`;
          }
        }
        else if (commandLower.startsWith('go_to_page:')) {
          const pageNum = parseInt(commandLower.split(':')[1]);
          if (isNaN(pageNum) || pageNum < 1 || pageNum > documentStore.totalPages) {
            return `Invalid page number. Please specify a page between 1 and ${documentStore.totalPages}.`;
          }
          documentStore.currentPageIndex = pageNum - 1;
          const currentPage = documentStore.documentPages[documentStore.currentPageIndex];
          return `[Page ${documentStore.currentPageIndex + 1}/${documentStore.totalPages} of "${documentStore.documentName}"]\n\n${currentPage.pageContent}`;
        }
        else if (commandLower === 'document_summary') {
          return `Document Information:
  - Name: ${documentStore.documentName}
  - Total Pages: ${documentStore.totalPages}
  - Current Page: ${documentStore.currentPageIndex + 1}
  - Format: ${path.extname(documentStore.documentName).substring(1).toUpperCase()}`;
        }
        else if (commandLower === 'page_count') {
          return `The document "${documentStore.documentName}" contains ${documentStore.totalPages} pages.`;
        }
        else {
          return `Unknown command: "${command}". Valid commands are: 'read_current_page', 'next_page', 'previous_page', 'go_to_page:{number}', 'document_summary', 'page_count'.`;
        }
      } catch (error) {
        console.error('Error reading page:', error);
        return `Error reading page: ${error.message}`;
      }
    }
  }
  
  // Custom voice control tool
  class VoiceControlTool extends Tool {
    name = "voice_control";
    description = "Control voice settings for Zoti. Input should be one of: 'list_voices', 'set_voice:{voice_name}', 'enable_voice', 'disable_voice', 'voice_status'.";
  
    async _call(command: string): Promise<string> {
      const commandLower = command.toLowerCase();
      
      try {
        // List available voices
        if (commandLower === 'list_voices') {
          return `Available voices:
  Male voices: ${voiceConfig.availableVoices.male.join(', ')}
  Female voices: ${voiceConfig.availableVoices.female.join(', ')}
  Currently selected: ${voiceConfig.selectedVoice}`;
        }
        // Set a specific voice
        else if (commandLower.startsWith('set_voice:')) {
          const voiceName = command.split(':')[1].trim();
          const allVoices = [...voiceConfig.availableVoices.male, ...voiceConfig.availableVoices.female];
          
          if (allVoices.includes(voiceName)) {
            voiceConfig.selectedVoice = voiceName;
            return `Voice set to ${voiceName}.`;
          } else {
            return `Voice "${voiceName}" not found. Available voices: ${allVoices.join(', ')}`;
          }
        }
        // Enable voice
        else if (commandLower === 'enable_voice') {
          voiceConfig.enabled = true;
          return 'Voice output enabled.';
        }
        // Disable voice
        else if (commandLower === 'disable_voice') {
          voiceConfig.enabled = false;
          return 'Voice output disabled.';
        }
        // Get voice status
        else if (commandLower === 'voice_status') {
          return `Voice output is currently ${voiceConfig.enabled ? 'enabled' : 'disabled'}.
  Selected voice: ${voiceConfig.selectedVoice}`;
        }
        else {
          return `Unknown command: "${command}". Valid commands are: 'list_voices', 'set_voice:{voice_name}', 'enable_voice', 'disable_voice', 'voice_status'.`;
        }
      } catch (error) {
        console.error('Error in voice control:', error);
        return `Error in voice control: ${error.message}`;
      }
    }
  }
  
  // Define the tools for the agent to use
  const tools = [
    new TavilySearchResults({ maxResults: 3, includeRawContent: true, includeAnswer: true }),
    new DocumentLoadTool(),
    new PageReaderTool(),
    new VoiceControlTool()
  ];
  const toolNode = new ToolNode(tools);
  
  // Enhanced system message for our Zoti document teaching agent
  const systemMessage = new SystemMessage(
    `# 📚 Welcome to Zoti School Slides Teacher! 

I'll be your teacher I am specialized in helping you understand school slides step by step. I'm designed to:

RULES:
1. YOU MUST TEACH THE STUDENT OR USERS AND ACT LIKE A PROFESSIONAL TEACHER,
2. YOU MUST MAKE SURE THEY UNDERSTAND THE CONTENT OF EACH PAGE BEFORE GOING TO THE NEXT PAGE
3. MAINATIN A PRFESSIONAL TONE ALWAYS
4. IF YOU DON'T KNOW THE ANSWER JUST SAY YOU DON'T KNOW DO NOT MAKE UP AN ANSWER
5. YOU MUST READ THE FULL PAGE OF THE SLIDE, BREAK IT DOWN AND EXPLAIN IT IN DETAIL
6. YOU MUST BE ABLE TO ANSWER SPECIFIC QUESTIONS ABOUT THE CONTENT OF THE SLIDES
7. YOU MUST BE ABLE TO SUMMARIZE THE CONTENT OF THE SLIDES


## 🔍 Document Analysis Capabilities:
- Follow all the rules above
- Ask for student's name at the beginning of the conversationa and use it to address the student throughout the conversation.
- Analyze documents comprehensively and teach it to the user like a professional teacher and a student
- Navigate through documents page by page, explaining content in detail
- Break down complex information into understandable lessons
- Highlight key concepts, definitions, and important passages
- Connect ideas across different parts of the document
- Provide summaries and contextual explanations

## 📋 Teaching Approach:
- You must read each file and understand the content, if you must, then search the internet for more information,
- You must be able to answer specific questions about the content of the slides,
- Guide you systematically through document content at your preferred pace
- Explain technical terminology and difficult concepts
- Answer specific questions about any part of the document
- Identify the main themes, arguments, and supporting evidence
- Relate document content to broader contexts when helpful
- Adapt my teaching style to your learning preferences

## 🛠️ Available Tools:
1. Web Search: I can search the internet to provide additional context for document content
2. Document Loader: I can load documents for analysis (use document_loader tool)
3. Page Navigator: I can read documents page by page with contextual explanations (use page_reader tool)

I'm committed to being your patient, thorough, and insightful document guide. Let me know how I can help you understand your documents better!`
  );
  
  // Create Groq model using ChatGroq with Groq's API endpoint
  // const groqModel = new ChatGroq({
  //   modelName: "mixtral-8x7b-32768", 
  //   temperature: 0.2,
  //   apiKey: process.env.GROQ_API_KEY
  // }).bindTools(tools);

  const model = new ChatGoogleGenerativeAI({
    modelName: "gemini-1.5-pro",
    maxOutputTokens: 2048,
    temperature: 0.2,
    apiKey: process.env.GOOGLE_API_KEY
  }).bindTools(tools);
  
  // Define the function that determines whether to continue or not
  function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
    const lastMessage = messages[messages.length - 1] as AIMessage;
  
    // If the LLM makes a tool call, then we route to the "tools" node
    if (lastMessage.tool_calls?.length) {
      return "tools";
    }
    // Otherwise, we stop (reply to the user) using the special "__end__" node
    return "__end__";
  }
  
  // Define the function that calls the model
  async function callModel(state: typeof MessagesAnnotation.State) {
    try {
      const response = await model.invoke(state.messages);
      return { messages: [response] };
    } catch (error) {
      console.error('Error in callModel:', error);
      // Return a fallback AI message if there's an error
      return {
        messages: [
          new AIMessage(
            "I'm having trouble connecting to my AI service right now. Please check your API keys in the .env file and try again."
          )
        ]
      };
    }
  }
  
  // Define a new graph
  const workflow = new StateGraph(MessagesAnnotation)
    .addNode("agent", callModel)
    .addEdge("__start__", "agent") // __start__ is a special name for the entrypoint
    .addNode("tools", toolNode)
    .addEdge("tools", "agent")
    .addConditionalEdges("agent", shouldContinue);
  
  // Compile the graph into a LangChain Runnable
  const app = workflow.compile();
  
  // Function to get user input
  function askQuestion(): Promise<string> {
    return new Promise((resolve) => {
      rl.question('👤 You: ', (answer) => {
        resolve(answer);
      });
    });
  }
  
  // Store the conversation state
  let messages = [systemMessage];
  const threadId = "zoti-document-teacher-" + Date.now();
  
  // Main chat loop
  async function startChat() {
    console.log('\n📚 Welcome to Zoti Document Teacher!');
    console.log('Your AI teaching assistant for understanding documents page by page');
    console.log('------------------------------------------------------------------');
    console.log('📄 Document Commands: ');
    console.log('  - To load a document: "Please load [file path]"');
    console.log('  - To navigate: "next page", "previous page", "go to page 5"');
    console.log('🔊 Voice Commands:');
    console.log('  - To list voices: "list voices"');
    console.log('  - To set a voice: "set voice:Matthew" (or any other available voice)');
    console.log('  - To enable/disable voice: "enable voice" or "disable voice"');
    console.log('❌ To exit: "exit" or "quit"');
    console.log('------------------------------------------------------------------');
    
    // Check if AWS credentials are available for voice features
    if (!pollyClient) {
      console.log('⚠️ Voice features are disabled because AWS credentials are missing.');
      console.log('  Add AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION to your .env file to enable voice.');
      voiceConfig.enabled = false;
    } else {
      // Test audio playback
      console.log('🔊 Testing voice functionality...');
      const testText = "Voice system initialized and ready.";
      await synthesizeSpeech(testText, voiceConfig.selectedVoice);
    }
    
    while (true) {
      const userInput = await askQuestion();
      
      if (userInput.toLowerCase() === 'exit' || userInput.toLowerCase() === 'quit') {
        console.log('👋 Thanks for learning with Zoti Document Teacher! Goodbye!');
        rl.close();
        break;
      }
      
      // Handle voice commands directly for faster response
      if (userInput.toLowerCase() === 'list voices') {
        console.log(`📚 Zoti: Available voices:
  Male voices: ${voiceConfig.availableVoices.male.join(', ')}
  Female voices: ${voiceConfig.availableVoices.female.join(', ')}
  Currently selected: ${voiceConfig.selectedVoice}`);
        continue;
      } else if (userInput.toLowerCase().startsWith('set voice:')) {
        const voiceName = userInput.split(':')[1].trim();
        const allVoices = [...voiceConfig.availableVoices.male, ...voiceConfig.availableVoices.female];
        
        if (allVoices.includes(voiceName)) {
          voiceConfig.selectedVoice = voiceName;
          console.log(`📚 Zoti: Voice set to ${voiceName}.`);
        } else {
          console.log(`📚 Zoti: Voice "${voiceName}" not found. Available voices: ${allVoices.join(', ')}`);
        }
        continue;
      } else if (userInput.toLowerCase() === 'enable voice') {
        voiceConfig.enabled = true;
        console.log('📚 Zoti: Voice output enabled.');
        continue;
      } else if (userInput.toLowerCase() === 'disable voice') {
        voiceConfig.enabled = false;
        console.log('📚 Zoti: Voice output disabled.');
        continue;
      } else if (userInput.toLowerCase() === 'voice status') {
        console.log(`📚 Zoti: Voice output is currently ${voiceConfig.enabled ? 'enabled' : 'disabled'}.
  Selected voice: ${voiceConfig.selectedVoice}`);
        continue;
      }
      
      // Add user message to the conversation
      messages.push(new HumanMessage(userInput));
      
      try {
        // Generate a response using our agent
        const result = await app.invoke(
          { messages },
          { configurable: { thread_id: threadId } }
        );
        
        // Update messages with the latest state
        messages = result.messages;
        
        // Display the AI's response
        const aiResponse = messages[messages.length - 1];
        console.log(`📚 Zoti: ${aiResponse.content}`);
        
        // Speak the response if voice is enabled
        if (pollyClient && voiceConfig.enabled) {
          // Extract the content string from the response
          const textToSpeak = typeof aiResponse.content === 'string' 
            ? aiResponse.content 
            : JSON.stringify(aiResponse.content);
          
          await synthesizeSpeech(textToSpeak, voiceConfig.selectedVoice);
        }
      } catch (error) {
        console.error('❌ Error:', error);
        console.log('📚 Zoti: Sorry, I encountered an error. Let\'s try again.');
        
        // Don't add the error to the conversation history
        messages = messages.slice(0, -1);
      }
    }
  }
  
  // Start the chat
  startChat();
