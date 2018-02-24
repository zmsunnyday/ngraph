// ----------------------------------------------------------------------------
// Copyright 2018 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <iostream>
#include <unordered_set>

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/CodeGen/ObjectFilePCHContainerOperations.h>
#include <clang/Driver/DriverDiagnostic.h>
#include <clang/Driver/Options.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/FrontendAction.h>
#include <clang/Frontend/FrontendActions.h>
#include <clang/Frontend/FrontendDiagnostic.h>
#include <clang/Frontend/TextDiagnosticBuffer.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/FrontendTool/Utils.h>
#include <clang/Lex/Preprocessor.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/ADT/Statistic.h>
#include <llvm/ExecutionEngine/MCJIT.h> // forces JIT to link in
#include <llvm/IR/Module.h>
#include <llvm/LinkAllPasses.h>
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/ManagedStatic.h>
#include <llvm/Support/Signals.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/Timer.h>
#include <llvm/Support/raw_ostream.h>

#include "compiler.hpp"
#include "file_util.hpp"
#include "util.hpp"

using namespace clang;
using namespace llvm;
using namespace llvm::opt;
using namespace std;

class Find_Includes : public PPCallbacks
{
public:
    Find_Includes(unordered_map<string, string>& include_files)
        : m_include_files(include_files)
    {
    }

    void InclusionDirective(SourceLocation hash_loc,
                            const Token& include_token,
                            StringRef file_name,
                            bool is_angled,
                            CharSourceRange filename_range,
                            const FileEntry* file,
                            StringRef search_path,
                            StringRef relative_path,
                            const clang::Module* imported) override
    {
        m_include_files.insert({relative_path.str(), search_path.str()});
    }
    unordered_map<string, string>& m_include_files;
};

class FindNamedClassVisitor : public RecursiveASTVisitor<FindNamedClassVisitor>
{
public:
    FindNamedClassVisitor(unordered_map<string, string>& files)
        : files_encountered(files)
    {
    }

    unordered_map<string, string>& files_encountered;
};

class FindNamedClassConsumer : public clang::ASTConsumer
{
public:
    FindNamedClassConsumer(unordered_map<string, string>& files)
        : m_visitor(files)
    {
    }
    virtual void HandleTranslationUnit(clang::ASTContext& Context)
    {
        // Traversing the translation unit decl via a RecursiveASTVisitor
        // will visit all nodes in the AST.
        m_visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }

private:
    // A RecursiveASTVisitor implementation.
    FindNamedClassVisitor m_visitor;
};

class FindNamedClassAction : public clang::ASTFrontendAction
{
public:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& Compiler,
                                                          llvm::StringRef InFile) override
    {
        return std::unique_ptr<clang::ASTConsumer>(new FindNamedClassConsumer(files_encountered));
    }
    unordered_map<string, string> files_encountered;

    bool BeginSourceFileAction(CompilerInstance& ci) override
    {
        std::unique_ptr<Find_Includes> find_includes_callback(new Find_Includes(m_include_files));

        Preprocessor& pp = ci.getPreprocessor();
        pp.addPPCallbacks(std::move(find_includes_callback));

        return true;
    }

    unordered_map<string, string> m_include_files;
};

HeaderInfo Compiler::collect_headers(const string& source)
{
    InitializeNativeTarget();
    LLVMInitializeNativeAsmPrinter();
    LLVMInitializeNativeAsmParser();

    // Prepare compilation arguments
    vector<const char*> args;
    string source_name = "code.cpp";
    args.push_back(source_name.c_str());

    // Prepare DiagnosticEngine
    string output;
    raw_string_ostream raw_out(output);
    IntrusiveRefCntPtr<DiagnosticOptions> diag_options = new DiagnosticOptions();
    diag_options->ErrorLimit = 20;
    IntrusiveRefCntPtr<DiagnosticIDs> diag_id(new DiagnosticIDs());
    DiagnosticsEngine diag_engine(diag_id, &*diag_options);

    // Create and initialize CompilerInstance
    m_compiler = std::unique_ptr<CompilerInstance>(new CompilerInstance());
    DiagnosticConsumer* diag_consumer;
    diag_consumer = new IgnoringDiagConsumer();
    m_compiler->createDiagnostics(diag_consumer);

    // Initialize CompilerInvocation
    CompilerInvocation::CreateFromArgs(
        m_compiler->getInvocation(), &args[0], &args[0] + args.size(), diag_engine);

    configure_search_path();

    // Language options
    // These are the C++ features needed to compile ngraph headers
    // and any dependencies like Eigen
    auto language_options = m_compiler->getInvocation().getLangOpts();
    language_options->CPlusPlus = 1;
    language_options->CPlusPlus11 = 1;
    language_options->Bool = 1;
    language_options->Exceptions = 1;
    language_options->CXXExceptions = 1;
    language_options->WChar = 1;
    language_options->RTTI = 1;
    // Enable OpenMP for Eigen
    language_options->OpenMP = 1;
    language_options->OpenMPUseTLS = 1;

    // CodeGen options
    auto& codegen_options = m_compiler->getInvocation().getCodeGenOpts();
    codegen_options.OptimizationLevel = 0;
    codegen_options.RelocationModel = "static";
    // codegen_options.CodeModel = "medium";
    codegen_options.ThreadModel = "posix";
    codegen_options.FloatABI = "hard";
    codegen_options.OmitLeafFramePointer = 1;
    codegen_options.VectorizeLoop = 1;
    codegen_options.VectorizeSLP = 1;
    codegen_options.CXAAtExit = 1;

    // Map code filename to a memoryBuffer
    StringRef source_ref(source);
    unique_ptr<MemoryBuffer> buffer = MemoryBuffer::getMemBufferCopy(source_ref);
    PreprocessorOptions& preprocessor_options = m_compiler->getInvocation().getPreprocessorOpts();
    preprocessor_options.RemappedFileBuffers.push_back({source_name, buffer.get()});

    // Create and execute action
    std::unique_ptr<clang::FrontendAction> m_compiler_action;
    FindNamedClassAction* action = new FindNamedClassAction();
    m_compiler_action.reset(action);
    if (m_compiler->ExecuteAction(*action) == true)
    {
    }
    buffer.release();

    for (const pair<string, string>& include : action->m_include_files)
    {
        cout << "include: " << include.first << ", " << include.second << endl;
    }

    preprocessor_options.RemappedFileBuffers.pop_back();

    HeaderInfo rc;
    // rc.headers.insert(
    //     rc.headers.begin(), action->files_encountered.begin(), action->files_encountered.end());
    // rc.search_paths = m_search_path_list;

    return rc;
}

void Compiler::configure_search_path()
{
    // Add base toolchain-supplied header paths
    // Ideally one would use the Linux toolchain definition in clang/lib/Driver/ToolChains.h
    // But that's a private header and isn't part of the public libclang API
    // Instead of re-implementing all of that functionality in a custom toolchain
    // just hardcode the paths relevant to frequently used build/test machines for now
    add_header_search_path(CLANG_BUILTIN_HEADERS_PATH);
    add_header_search_path("/usr/include/x86_64-linux-gnu");
    add_header_search_path("/usr/include");

    // Search for headers in
    //    /usr/include/x86_64-linux-gnu/c++/N.N
    //    /usr/include/c++/N.N
    // and add them to the header search path

    file_util::iterate_files("/usr/include/x86_64-linux-gnu/c++/",
                             [&](const std::string& file, bool is_dir) {
                                 if (is_dir)
                                 {
                                     string dir_name = file_util::get_file_name(file);
                                     if (is_version_number(dir_name))
                                     {
                                         add_header_search_path(file);
                                     }
                                 }
                             });

    file_util::iterate_files("/usr/include/c++/", [&](const std::string& file, bool is_dir) {
        if (is_dir)
        {
            string dir_name = file_util::get_file_name(file);
            if (is_version_number(dir_name))
            {
                add_header_search_path(file);
            }
        }
    });

    add_header_search_path(EIGEN_HEADERS_PATH);
    add_header_search_path(TBB_HEADERS_PATH);
    add_header_search_path(NGRAPH_HEADERS_PATH);
    // add_header_search_path(INSTALLED_HEADERS_PATH);
}

void Compiler::add_header_search_path(const string& path)
{
    HeaderSearchOptions& hso = m_compiler->getInvocation().getHeaderSearchOpts();
    hso.AddPath(path, clang::frontend::System, false, false);
    m_search_path_list.push_back(path);
}

bool Compiler::is_version_number(const string& path)
{
    bool rc = true;
    vector<string> tokens = split(path, '.');
    for (string s : tokens)
    {
        for (char c : s)
        {
            if (!isdigit(c))
            {
                rc = false;
            }
        }
    }
    return rc;
}
