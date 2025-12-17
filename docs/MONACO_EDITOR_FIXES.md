# Monaco Editor & Gemini Code Reading/Writing Fixes

## Issues Found

1. **maxOutputTokens was too low**: Set to 200, which was insufficient for code responses containing `[CODE_START]...[CODE_END]` markers
2. **Missing clear instructions**: Gemini wasn't explicitly told to read and analyze existing code before responding
3. **No emphasis on code visibility**: The prompt didn't emphasize that Gemini can see the current code in the editor

## Fixes Applied

### 1. Increased maxOutputTokens (app/gemini_client.py)
- **Lines 140, 272**: Changed from 200 to 800 tokens
- This allows Gemini to generate longer responses including complete code blocks with proper markers

### 2. Enhanced Code Reading Instructions
Added explicit instructions to Gemini in the coding phase prompt:
- **Code visibility**: "You can SEE the current code in the editor (shown above in 'CURRENT CODE IN EDITOR')"
- **Always read first**: "ALWAYS read and understand their existing code before responding to ANY question or comment"
- **Reference capability**: "You can reference specific parts of their code in your responses"
- **Analysis requirements**: "ALWAYS look at their current code in the 'CURRENT CODE IN EDITOR' section above"

### 3. Improved Code Writing Instructions
- **Before writing**: "CRITICAL: Before writing code, ALWAYS look at what they already have"
- **Modification capability**: "You can modify their existing code or provide a complete replacement"
- **Better examples**: Added more detailed examples showing how to write code using `[CODE_START]...[CODE_END]` markers

## How It Works Now

1. **Code Reading**: 
   - When the user speaks, the current code from the Monaco editor is sent to Gemini
   - Gemini is explicitly told to read and understand this code
   - Gemini can reference specific parts in responses (e.g., "I see you've started with a hash map approach")

2. **Code Writing**:
   - When Gemini needs to write code, it uses the `[CODE_START]...code...[CODE_END]` format
   - The JavaScript `extractGeminiCode()` function extracts this and updates the Monaco editor
   - The user sees a system message "📝 Gemini updated the code editor"

3. **Token Limit**:
   - Increased from 200 to 800 tokens allows for complete code solutions
   - Still keeps responses focused (not essay-length)

## Testing

To test the fixes:
1. Start a coding interview
2. Type some code in the Monaco editor
3. Ask Gemini: "What do you think of my code?" (Gemini should analyze it)
4. Ask Gemini: "Can you write a solution for me?" (Gemini should use [CODE_START]...[CODE_END] markers)

## Files Modified

- `app/gemini_client.py`: 
  - Lines 140, 272: maxOutputTokens increased to 800
  - Lines 555-590: Enhanced code reading/writing instructions
  - Multiple locations: Added explicit code visibility instructions

## Status

✅ Gemini can now read code from the Monaco editor
✅ Gemini can now write code using proper markers
✅ Token limit increased to support code generation
✅ Clear instructions ensure Gemini understands its capabilities 